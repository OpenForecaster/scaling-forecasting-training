#!/bin/bash

# Fix CUDA environment issues for job submission
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Fix Ray environment issues
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Fix serialization issues
# export PYTHONPATH="${PYTHONPATH}:/lustre/home/nchandak/forecasting/libraries/verl"
export RAY_DISABLE_IMPORT_WARNING=1

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
NUM_GPUS=8 # Use only first 4 GPUs (0,1,2,3) for training
echo "Number of available GPUs: $NUM_GPUS"

# Set CUDA_VISIBLE_DEVICES to use GPUs 0-3 for training and 7 for verifier
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clip_ratio_low=0.2
clip_ratio_high=0.28
PROJECT_NAME="rl-data66k-withbinary2k"

# MODEL_NAME="Qwen3-4B-Instruct-2507" # -Thinking-2507"
MODEL_NAME="Qwen3-8B"
MODEL_PATH="/fast/nchandak/models/$MODEL_NAME"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
TRAINER_EXPERIMENT_NAME="$MODEL_NAME-$MAX_PROMPT_LENGTH-$MAX_RESPONSE_LENGTH-acc-kl0.005"

taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
reward_model.enable=True \
reward_model.model.path=/fast/nchandak/models/Qwen3-4B \
reward_model.strategy=matcher \
reward_model.reward_manager=naive \
reward_model.micro_batch_size=0 \
data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/data70k-withbinary2k/combined_non_numeric_all_train.jsonl \
data.val_files=/fast/nchandak/forecasting/datasets/verl/freeform/data70k-withbinary2k/combined_non_numeric_all_validation.jsonl \
data.train_batch_size=256 \
data.val_batch_size=512 \
data.shuffle=False \
data.filter_overlong_prompts=True \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=5e-6 \
actor_rollout_ref.actor.optim.warmup_style=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.005 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
actor_rollout_ref.rollout.val_kwargs.n=1 \
trainer.critic_warmup=0 \
trainer.logger=['console','wandb'] \
++trainer.val_before_train=False \
trainer.n_gpus_per_node=$NUM_GPUS \
trainer.nnodes=1 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.save_freq=100 \
trainer.test_freq=20 \
trainer.total_epochs=10 \
trainer.default_local_dir="/fast/nchandak/forecasting/training/verl/checkpoints/${PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}" \
+reward_manager='prime'