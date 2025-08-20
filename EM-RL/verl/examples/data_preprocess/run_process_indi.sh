export HF_HOME="PATHS"
export HF_DATASETS_CACHE="PATHS"
export HF_TOKEN="TOKEN"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python post_process_indivalue.py