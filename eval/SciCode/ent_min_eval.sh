
export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export HF_ALLOW_CODE_EVAL="1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=Qwen/Qwen2.5-32B-Instruct
MODE=normal # Options: normal, min_entropy, self_refinement
ITER=3


for threshold in 0.05 0.1 0.3 0.5; do
    for kl_weight in 0.001 0.01; do
        for learning_rate in 0.01 0.1; do
            for n_grad_steps in 3 5; do
                MODEL=Qwen/Qwen2.5-7B-Instruct
                echo "Running $MODEL in temp_decrease mode with threshold $threshold, kl_weight $kl_weight, learning_rate $learning_rate, and n_grad_steps $n_grad_steps..."
                hyperparameters="{\"threshold\": $threshold, \"kl_weight\": $kl_weight, \"learning_rate\": $learning_rate, \"n_grad_steps\": $n_grad_steps}"
                python eval/scripts/gencode.py --model openai/$MODEL \
                    --split test --mode min_entropy --with-background --temperature 0.1 --hyperparameters "$hyperparameters"

                sleep 10

                python eval/scripts/test_generated_code.py --model $MODEL --mode min_entropy --with-background --hyperparameters "$hyperparameters"
            done
        done
    done
done