#!/bin/bash

################ Arguments #################
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
TASK=ugphysics

tmax=1
tmin=0.01
max_iter=100
tol=0.1
target_ratio=0.85
threshold=0.00001
############################################

HYPERPARAMETERS="{\"tmax\": $tmax, \"tmin\": $tmin, \"max_iter\": $max_iter, \"tol\": $tol, \"target_ratio\": $target_ratio, \"threshold\": $threshold}"
bash ./eval/run.sh \
    --model_ckpt $MODEL_NAME_OR_PATH \
    --mode adaptive_temp \
    --temp 0.1 \
    --task $TASK \
    --hyperparameters "$HYPERPARAMETERS"