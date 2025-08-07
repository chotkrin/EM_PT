# vllm serve Qwen/Qwen2.5-72B-Instruct \
#     --dtype auto \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.7 \
#     --trust-remote-code \
#     --api-key token-abc123
export VLLM_USE_V1=0

vllm serve Qwen/Qwen2.5-32B-Instruct \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --logits-processor-pattern scicode.utils.min_entropy.AdaptiveTemperatureProcessor \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --api-key token-abc123

# vllm serve meta-llama/Llama-3.1-8B-Instruct \
#     --dtype auto \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.7 \
#     --trust-remote-code \
#     --api-key token-abc123

# --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \