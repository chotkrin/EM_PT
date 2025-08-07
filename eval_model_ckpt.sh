#!/bin/bash

export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export HF_ALLOW_CODE_EVAL="1"
export VLLM_USE_V1=0


TEMP=0.1
TASK=math

hyperparameters='{"threshold": 0.3, "kl_weight": 0.05, "learning_rate": 0.1, "n_grad_steps": 3, "temp": 0.1}'
# hyperparameters='{"threshold": 0.1, "kl_weight": 0.001, "learning_rate": 0.1, "n_grad_steps": 3, "temp": 0.1, "temp_decrease": false}'
# hyperparameters='{"threshold": 0.3, "kl_weight": 0.001, "learning_rate": 0.1, "n_grad_steps": 3, "temp": 0.1, "temp_decrease": false}'
# hyperparameters='{"threshold": 0.3, "kl_weight": 0.0, "learning_rate": 0.1, "n_grad_steps": 5, "temp": 0.1, "temp_decrease": false}'
hyperparameters_temp_decrease='{"tmax": 1, "tmin": 0.01, "max_iter": 100, "tol": 0.1, "target_ratio": 0.85, "threshold": 0.00001}'

###################### EM-INF ######################
# BASE_PATH=Qwen/Qwen2.5-7B-Instruct
# bash ./eval/run.sh \
#     --model_ckpt $BASE_PATH \
#     --mode min_entropy \
#     --temp $TEMP \
#     --task $TASK \
#     --min_ent_params "$hyperparameters" \
#     --num_processes 16

##################### Normal Evaluation ######################
BASE_PATH=Qwen/Qwen2.5-7B-Instruct
bash ./eval/run.sh --model_ckpt $BASE_PATH --mode normal --temp $TEMP --task $TASK

###################### Self-Consistency Evaluation ######################
# BASE_PATH=Qwen/Qwen2.5-7B-Instruct
# # BASE_PATH=allenai/Llama-3.1-Tulu-3-8B
# bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh --model_ckpt $BASE_PATH --mode self_consistency --temp $TEMP --task $TASK

###################### Self-Refinement Evaluation ######################
# BASE_PATH=Qwen/Qwen2.5-7B-Instruct
# # BASE_PATH=meta-llama/Llama-3.1-8B-Instruct
# bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh --model_ckpt $BASE_PATH --mode ice_self_refinement --temp $TEMP --task $TASK --n_trajs 3

###################### ICE + Self-Consistency Evaluation ######################
# BASE_PATH=Qwen/Qwen2.5-7B-Instruct
# # BASE_PATH=allenai/Llama-3.1-Tulu-3-8B
# bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh --model_ckpt $BASE_PATH --mode ice_self_consistency --temp $TEMP --task $TASK --min_ent_params "$hyperparameters" --num_processes 16

###################### Min Entropy Evaluation ######################
# BASE_PATH=Qwen/Qwen2.5-Math-7B-Instruct
# for ratio in 0.05 0.1 0.25 0.5 0.75 0.85 0.95; do
#     for threshold in 0.00001 0.001 0.1 0.3; do
#         hyperparameters="{\"tmax\": 1, \"tmin\": 0.01, \"max_iter\": 100, \"tol\": 0.1, \"target_ratio\": $ratio, \"threshold\": $threshold, \"temp_decrease\": true}"
#         echo "Running temp_decrease with target_ratio=$ratio and threshold=$threshold on model $BASE_PATH"
#         bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh \
#             --model_ckpt $BASE_PATH \
#             --mode min_entropy \
#             --temp $TEMP \
#             --task $TASK \
#             --min_ent_params "$hyperparameters" \
#             --num_processes 16
#     done
# done

# BASE_PATH=Qwen/Qwen2.5-7B-Instruct
# # BASE_PATH=meta-llama/Llama-3.1-8B-Instruct
# bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh --model_ckpt $BASE_PATH --mode min_entropy --temp $TEMP --task $TASK --min_ent_params "$hyperparameters_temp_decrease" --num_processes 16
###################### Multiple model with same format nameing ######################
# BASE_PATH=/work/nvme/bebu/shared_ckpts/ckpts_verl/rloo_7b_math_qwen_soft_em_prob_fix_with_consistency/global_step_20/actor/huggingface
# for iter in 20 40 60; do
#     MODEL_CKPT=/work/nvme/bebu/shared_ckpts/ckpts_verl/rloo_7b_math_llama/global_step_${iter}/actor/huggingface
#     echo "Evaluating iter_$iter checkpoint: $MODEL_CKPT"
#     bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh $MODEL_CKPT normal $TEMP $TASK
# done

# BASE_PATH=/work/nvme/bebu/shared_ckpts/ckpts_verl/rloo_7b_math_qwen_soft_em_prob_fix_with_consistency/global_step_20/actor/huggingface
# for iter in 20 40 60 70; do
#     MODEL_CKPT=/work/nvme/bebu/shared_ckpts/ckpts_verl/rloo_7b_math_LLAMA_soft_em_prob_no_cons_log_prob_avg/global_step_${iter}/actor/huggingface
#     echo "Evaluating iter_$iter checkpoint: $MODEL_CKPT"
#     bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh $MODEL_CKPT normal $TEMP $TASK
# done

###################### Coding benchmark ######################
# BASE_PATH=/work/nvme/bebu/shared_ckpts/ckpts_verl/prime_7b_coding_qwen_soft_em_prob_no_cons_log_prob_avg/global_step_15/actor/huggingface
# bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh $BASE_PATH normal $TEMP $TASK

# BASE_PATH=Qwen/Qwen2.5-7B-Instruct

# for threshold in 0.1 0.3; do
#   for kl_weight in 0 0.01; do
#     for learning_rate in 0.01; do
#       for n_grad_steps in 1 3 7; do
#         for temp_param in 0.1 0.3 0.7; do
#           hyperparameters="{\"threshold\": $threshold, \"kl_weight\": $kl_weight, \"learning_rate\": $learning_rate, \"n_grad_steps\": $n_grad_steps, \"temp\": $temp_param}"
#           echo "Running with: $hyperparameters"
#           bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh \
#             --model_ckpt $BASE_PATH \
#             --mode ice_self_consistency \
#             --temp $TEMP \
#             --task $TASK \
#             --min_ent_params "$hyperparameters" \
#             --num_processes 16 \
#             --n_trajs 16
#         done
#       done
#     done
#   done
# done

# BASE_PATH=Qwen/Qwen2.5-7B-Instruct
# for n_trajs in 4 8 16; do
#   bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh \
#     --model_ckpt $BASE_PATH \
#     --mode ice_self_consistency \
#     --temp $TEMP \
#     --task aime \
#     --n_trajs $n_trajs \
#     --min_ent_params "$hyperparameters" \
#     --num_processes 16
# done

# BASE_PATH=Qwen/Qwen2.5-Math-7B-Instruct
# for threshold in 0.1 0.3 0.5 0.7 1.0; do
#   hyperparameters="{\"threshold\": $threshold, \"kl_weight\": 0.0, \"learning_rate\": 0.1, \"n_grad_steps\": 5, \"temp\": 0.1, \"temp_decrease\": true}"
#   echo "Running min_entropy with threshold=$threshold"
#   bash /work/nvme/bcaq/zzhang32/PRIME_inference_scaling/eval/run.sh \
#     --model_ckpt $BASE_PATH \
#     --mode min_entropy \
#     --temp $TEMP \
#     --task $TASK \
#     --min_ent_params "$hyperparameters" \
#     --num_processes 16
# done