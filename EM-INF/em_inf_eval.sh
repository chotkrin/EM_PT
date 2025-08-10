#!/bin/bash

################ Arguments #################
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
TEMP=0.1
TASK=ugphysics
NUM_PROCESSES=16

threshold=0.3
kl_weight=0.05
learning_rate=0.1
n_grad_steps=3
############################################

HYPERPARAMETERS="{\"threshold\": $threshold, \"kl_weight\": $kl_weight, \"learning_rate\": $learning_rate, \"n_grad_steps\": $n_grad_steps}"
bash ./eval/run.sh \
    --model_ckpt $MODEL_NAME_OR_PATH \
    --mode em_inf \
    --temp $TEMP \
    --task $TASK \
    --hyperparameters "$HYPERPARAMETERS" \
    --num_processes $NUM_PROCESSES