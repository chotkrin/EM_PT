# vllm serve Qwen/Qwen2.5-72B-Instruct \
#     --dtype auto \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.7 \
#     --trust-remote-code \
#     --api-key token-abc123

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --api-key token-abc123