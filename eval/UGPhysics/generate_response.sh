export HF_HOME="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/zzhang32/hf_cache/"
export HF_TOKEN=""
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real 

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

# module load cuda/12.4
export HF_ALLOW_CODE_EVAL="1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENAI_API_KEY=token-abc123
export OPENAI_BASE_URL=http://localhost:8000/v1

MODEL=Qwen/Qwen2.5-32B

SUB_LIST=(
    "SemiconductorPhysics"
)

for subject in "${SUB_LIST[@]}"; do
    echo "*********************************************"
    python codes/generate_open.py --model $MODEL --subject ${subject} --tensor_parallel_size 4
    echo "*********************************************"
done