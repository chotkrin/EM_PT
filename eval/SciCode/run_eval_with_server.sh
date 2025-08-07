#!/bin/bash

# Set up environment variables from eval_manual.sh
export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export HF_ALLOW_CODE_EVAL="1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

# Function to cleanup server on script exit
cleanup() {
    echo "Shutting down vLLM server..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
    fi
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Start vLLM server in background
echo "Starting vLLM server..."
vllm serve Qwen/Qwen2.5-32B-Instruct \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --logits-processor-pattern scicode.utils.min_entropy.AdaptiveTemperatureProcessor \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --api-key token-abc123 &

# Store the process ID for cleanup
VLLM_PID=$!

echo "vLLM server started with PID: $VLLM_PID"

# Function to check if server is ready
wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=60  # Wait up to 5 minutes
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        
        sleep 5
        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts - Server not ready yet..."
    done
    
    echo "Server failed to start within timeout period"
    return 1
}

# Wait for server to be ready
if ! wait_for_server; then
    echo "Failed to start vLLM server"
    exit 1
fi

# Run the evaluation loops (from eval_manual.sh)
echo "Starting evaluation..."

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 
do
    for target_threshold in 0.00001 0.001 0.1
    do
        MODEL=Qwen/Qwen2.5-32B-Instruct
        echo "Running $MODEL in temp_decrease mode with ratio $ratio and target_threshold $target_threshold..."
        hyperparameters="{\"tmax\": 1, \"tmin\": 0.01, \"max_iter\": 100, \"tol\": 0.1, \"target_ratio\": $ratio, \"target_threshold\": $target_threshold}"
        
        python eval/scripts/gencode.py --model openai/$MODEL \
            --split test --mode temp_decrease --temperature 0.1 --hyperparameters "$hyperparameters"

        sleep 10

        python eval/scripts/test_generated_code.py --model $MODEL --mode temp_decrease --hyperparameters "$hyperparameters"
    done
done

echo "Evaluation completed!"
# Server will be automatically shut down by the cleanup function