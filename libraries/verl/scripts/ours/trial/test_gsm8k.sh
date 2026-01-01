#!/bin/bash
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export VLLM_ATTENTION_BACKEND=XFORMERS

# /fast/rolmedo/models/qwen2.5-7b-it 
# 'qwen2.5-7b-datamix-train-brier'


# actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
# actor_rollout_ref.rollout.disable_chunked_prefill=True \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=True \

# data.max_prompt_length=1024 \
# data.max_response_length=7168 \
# actor_rollout_ref.rollout.enable_chunked_prefill=False \
# actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of available GPUs: $NUM_GPUS"


MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=2048
TRAINER_EXPERIMENT_NAME="qwen2.5-1.5b-gsm8k-512-2048"

taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=/fast/nchandak/forecasting/misc/gsm8k/train.parquet \
data.val_files=/fast/nchandak/forecasting/misc/gsm8k/test.parquet \
data.train_batch_size=512 \
data.val_batch_size=128 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
actor_rollout_ref.model.path=/fast/rolmedo/models/qwen2.5-1.5b-it \
actor_rollout_ref.actor.optim.lr=5e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.04 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.temperature=0.9 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.critic_warmup=0 \
trainer.logger=['console','wandb'] \
++trainer.val_before_train=True \
trainer.n_gpus_per_node=$NUM_GPUS \
trainer.nnodes=1 \
trainer.project_name='rl-test' \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.save_freq=-1 \
trainer.test_freq=2 \
trainer.total_epochs=15 \
trainer.default_local_dir="/fast/nchandak/forecasting/training/verl/checkpoints/${TRAINER_EXPERIMENT_NAME}" \
+reward_manager='prime'