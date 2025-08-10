#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL="1"
export VLLM_USE_V1=0

# Function to cleanup server on script exit
cleanup() {
    echo "Shutting down vLLM server..."
    if [ ! -z "$VLLM_PID" ]; then
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
    fi
    # Optionally clean up log files (comment out if you want to keep them)
    # rm -f "$LOG_DIR/vllm_server.log" "$LOG_DIR/vllm_server_error.log"
}

# Set trap to cleanup on script exit
trap cleanup EXIT

MODEL=Qwen/Qwen2.5-7B-Instruct
MODE=normal
TEMP=0.1

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    num_gpus=$(nvidia-smi --list-gpus | wc -l)
fi

# Define log directory
LOG_DIR="/work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode/vllm_server_logs"
mkdir -p "$LOG_DIR"

# Start vLLM server in background
echo "Starting vLLM server..."
echo "Server logs will be saved to $LOG_DIR/vllm_server.log and $LOG_DIR/vllm_server_error.log"
vllm serve $MODEL \
    --dtype auto \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --logits-processor-pattern scicode.utils.min_entropy.AdaptiveTemperatureProcessor \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --api-key token-abc123 > "$LOG_DIR/vllm_server.log" 2> "$LOG_DIR/vllm_server_error.log" &

# Store the process ID for cleanup
VLLM_PID=$!

echo "vLLM server started with PID: $VLLM_PID"

# Function to check if server is ready
wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=15  # Wait up to 5 minutes
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        
        sleep 20
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

python eval/scripts/gencode.py \
    --model openai/$MODEL \
    --split test \
    --mode $MODE \
    --temperature $TEMP \
    --prompt-dir /work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode/prompts \
    --output-dir /work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode/generated_code \
    --with-background

sleep 10

python eval/scripts/test_generated_code.py \
    --model $MODEL \
    --mode $MODE \
    --code-dir /work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode/generated_code \
    --log-dir /work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode/logs \
    --tmp-dir /work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode \
    --output-dir /work/nvme/bcaq/zzhang32/EM_PT/results/Qwen__Qwen2.5-7B-Instruct/normal/scicode \
    --hyperparameters "$hyperparameters"

echo "Evaluation completed!"
# Server will be automatically shut down by the cleanup function