
export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export HF_ALLOW_CODE_EVAL="1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# MODEL=Qwen/Qwen2.5-32B-Instruct
# MODE=normal # Options: normal, min_entropy, self_refinement
# ITER=3

# if [ "$MODE" = "normal" ]; then
#     export VLLM_USE_V1=0
#     echo "Running in $MODE mode..."
#     python eval/scripts/gencode.py --model openai/$MODEL \
#         --split test --mode "normal" --temperature 0.1 --with-background

#     sleep 10

#     python eval/scripts/test_generated_code.py --model $MODEL --with-background

# elif [ "$MODE" = "min_entropy" ]; then
#     echo "Running in $MODE mode..."
#     python eval/scripts/gencode.py --model openai/$MODEL \
#         --split test --mode "min_entropy" --temperature 0.1 --with-background

#     sleep 10

#     python eval/scripts/test_generated_code.py --model $MODEL --with-background

# elif [ "$MODE" = "self_refinement" ]; then
#     echo "Running in self_refinement mode for $ITER iterations..."
#     for (( i=0; i<$ITER; i++ ))
#     do
#       echo "Iteration $i..."
#       python eval/scripts/gencode.py --model openai/$MODEL \
#           --split test --mode $MODE --current_iter $i --temperature 0

#       sleep 10

#       python eval/scripts/test_generated_code.py --model $MODEL --current_iter $i --mode $MODE 
#     done
# else
#     echo "Unknown MODE: $MODE"
# fi

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 
do
    for target_threshold in 0.00001 0.001 0.1
    do
        export VLLM_USE_V1=0
        MODEL=Qwen/Qwen2.5-32B-Instruct
        echo "Running $MODEL in temp_decrease mode with ratio $ratio and target_threshold $target_threshold..."
        hyperparameters="{\"tmax\": 1, \"tmin\": 0.01, \"max_iter\": 100, \"tol\": 0.1, \"target_ratio\": $ratio, \"target_threshold\": $target_threshold}"
        python eval/scripts/gencode.py --model openai/$MODEL \
            --split test --mode temp_decrease --temperature 0.1 --hyperparameters "$hyperparameters"

        sleep 10

        python eval/scripts/test_generated_code.py --model $MODEL --mode temp_decrease --hyperparameters "$hyperparameters"
    done
done