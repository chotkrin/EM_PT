#!/bin/bash

################ Arguments #################
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
TEMP=0.1
TASK=scicode
############################################

bash ./eval/run.sh \
    --model_ckpt $MODEL_NAME_OR_PATH \
    --mode normal \
    --temp $TEMP \
    --task $TASK