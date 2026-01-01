#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Configure Ray to properly handle GPU allocation
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

export HYDRA_FULL_ERROR=1

# Configure Ray to allow GPU sharing (not isolation)
# export RAY_GPU_MEMORY_FRACTION=0.8
# export RAY_GPU_MEMORY_ALLOW_GROWTH=true

# /fast/rolmedo/models/qwen2.5-7b-it 
# 'qwen2.5-7b-datamix-train-brier'


# actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
# actor_rollout_ref.rollout.disable_chunked_prefill=True \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=True \

# reward_model.model.path=/fast/nchandak/models/general-verifier \

# data.max_prompt_length=1024 \
# data.max_response_length=7168 \
# actor_rollout_ref.rollout.enable_chunked_prefill=False \
# actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
# Get the number of available GPUs
NUM_GPUS=4 # Use only first 4 GPUs (0,1,2,3) for training
echo "Number of available GPUs: $NUM_GPUS"

# Set CUDA_VISIBLE_DEVICES to use GPUs 0-3 for training and 7 for verifier
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024
TRAINER_EXPERIMENT_NAME="qwen2.5-0.5b-gsm8k-matcher"

taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
reward_model.enable=True \
reward_model.model.path=/fast/nchandak/models/Qwen3-1.7B \
reward_model.strategy=matcher \
reward_model.reward_manager=naive \
reward_model.micro_batch_size=0 \
data.train_files=/fast/nchandak/forecasting/misc/random/train2.parquet \
data.val_files=/fast/nchandak/forecasting/misc/random/test2.parquet \
data.train_batch_size=256 \
data.val_batch_size=128 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
actor_rollout_ref.model.path=/fast/rolmedo/models/qwen2.5-3b-it \
actor_rollout_ref.actor.optim.lr=5e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.04 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=4 \
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