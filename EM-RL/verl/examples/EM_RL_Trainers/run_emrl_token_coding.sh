export HF_HOME="PATHS"
export HF_DATASETS_CACHE="PATHS"
export HF_TOKEN="HF_TOKEN"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files="train.parquet" \
    data.val_files="test.parquet" \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path='PRIME-RL/Eurus-2-7B-SFT' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ent_minimization' \
    trainer.experiment_name='prime_7b_coding_token' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=15 \
    trainer.test_freq=-1 \
    reward_model.reward_manager='emrl_token' \
    trainer.total_epochs=1 $@
