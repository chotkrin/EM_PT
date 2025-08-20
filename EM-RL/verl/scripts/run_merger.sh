export HF_HOME="PATHS"
export HF_DATASETS_CACHE="PATHS"
export HF_TOKEN="HF_TOKEN"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real



python model_merger.py --local_dir 'CKPT_PATH'
