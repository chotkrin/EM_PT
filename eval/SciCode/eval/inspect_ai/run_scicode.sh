export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

# module load cuda/12.4
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=Qwen/Qwen2.5-7B-Instruct
inspect eval scicode.py \
    --model vllm/$MODEL \
    --temperature 0 \
    -M gpu_memory_utilization=0.6 \
    -M tensor_parallel_size=4 \
    -T with_background=False \
    -T mode=normal

# MODEL=Qwen/Qwen2.5-32B-Instruct
# inspect eval scicode.py \
#     --model vllm/$MODEL \
#     -T with_background=False \
#     -T mode=normal

# MODEL=/work/nvme/bcaq/zzhang32/pu_learning/checkpoint/openrlhf-llama3-sft-mixture_ce-lab0.05-unlab0.95-onpolicy-task-grouping-pu-avg-rm_normal-ds_dpo/checkpoint
# inspect eval scicode.py \
#     --temperature 0 \
#     --model vllm/local \
#     -M model_path=$MODEL \
#     -M gpu_memory_utilization=0.7 \
#     -M tensor_parallel_size=4 \
#     -T with_background=True \
#     -T mode=normal