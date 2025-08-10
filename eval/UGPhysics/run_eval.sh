#!/bin/bash

MODE=normal
TEMP=0.1

python codes/generate_open.py --model $MODEL --subject SemiconductorPhysics --tensor_parallel_size 4 --mode temp_decrease --n_repeat 1 --temperature $TEMP --hyperparameters "$hyperparameters_temp_decrease"


# export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
# export HF_TOKEN=""
# export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

# export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

# # module load cuda/12.4
# export HF_ALLOW_CODE_EVAL="1"
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export OPENAI_API_KEY=token-abc123
# export OPENAI_BASE_URL=http://localhost:8000/v1

# MODEL=meta-llama/Llama-3.1-8B-Instruct
# hyperparameters="{\"threshold\": $threshold, \"kl_weight\": 0.0, \"learning_rate\": 0.1, \"n_grad_steps\": 5, \"temp\": 0.1, \"temp_decrease\": true}"

# SUB_LIST=(
#     "SemiconductorPhysics"
# )

# for subject in "${SUB_LIST[@]}"; do
#     echo "*********************************************"
#     python codes/generate_open.py --model $MODEL --subject ${subject} --tensor_parallel_size 4 --mode temp_decrease --n_repeat 1 --temperature 0.1 --min_ent_params "$hyperparameters_temp_decrease"
#     echo "*********************************************"
# done

# echo "Evaluating individual subject......"
# for subject in "${SUB_LIST[@]}"; do
#     echo "*********************************************"
#     python codes/eval.py --model_path ${MODEL} --subject ${subject} --mode temp_decrease --min_ent_params "$hyperparameters_temp_decrease"
#     echo "*********************************************"
# done

# echo "Evaluating all subjects......"
# python codes/eval.py --model_path ${MODEL} --subject all

# MODEL=Qwen/Qwen2.5-7B-Instruct

# SUB_LIST=(
#     "SemiconductorPhysics"
# )

# for threshold in 0.3 0.5 0.7; do
#     hyperparameters="{\"tmax\": 1, \"tmin\": 0.01, \"max_iter\": 100, \"tol\": 0.1, \"target_ratio\": $ratio, \"threshold\": $threshold}"
#     hyperparameters="{\"threshold\": $threshold, \"kl_weight\": 0.0, \"learning_rate\": 0.1, \"n_grad_steps\": 5, \"temp\": 0.1, \"temp_decrease\": true}"
#     for subject in "${SUB_LIST[@]}"; do
#         echo "*********************************************"
#         python codes/generate_open.py --model $MODEL --subject ${subject} --tensor_parallel_size 4 --mode min_entropy --n_repeat 1 --temperature 0.1 --min_ent_params "$hyperparameters"
#         echo "*********************************************"
#     done
# done

# MODEL=meta-llama/Llama-3.1-8B-Instruct
# SUB_LIST=(
#     "SemiconductorPhysics"
# )
# for ratio in 0.05 0.1 0.25 0.3 0.5 0.7; do
#     for threshold in 0.0000001 0.00000001; do
#         hyperparameters="{\"tmax\": 1, \"tmin\": 0.01, \"max_iter\": 100, \"tol\": 0.1, \"target_ratio\": $ratio, \"threshold\": $threshold}"
#         for subject in "${SUB_LIST[@]}"; do
#             echo "*********************************************"
#             python codes/generate_open.py --model $MODEL --subject ${subject} --tensor_parallel_size 4 --mode temp_decrease --n_repeat 1 --temperature 0.1 --min_ent_params "$hyperparameters"
#             echo "*********************************************"
            
#             echo "Evaluating individual subject......"
#             for subject in "${SUB_LIST[@]}"; do
#                 echo "*********************************************"
#                 python codes/eval.py --model_path ${MODEL} --subject ${subject} --mode temp_decrease --min_ent_params "$hyperparameters"
#                 echo "*********************************************"
#             done
#         done
#     done
# done