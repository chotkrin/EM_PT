#!/bin/bash
export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export HF_ALLOW_CODE_EVAL="1"
export VLLM_USE_V1=0

################ Arguments #################
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
TEMP=0.1
TASK=math
############################################

bash ./eval/run.sh \
    --model_ckpt $MODEL_NAME_OR_PATH \
    --mode normal \
    --temp $TEMP \
    --task $TASK