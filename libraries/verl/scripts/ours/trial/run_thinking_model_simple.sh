#!/bin/bash

# Simplified version for job submission
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Minimal Ray configuration
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_IMPORT_WARNING=1

export HYDRA_FULL_ERROR=1

NUM_GPUS=8
echo "Number of available GPUs: $NUM_GPUS"

clip_ratio_low=0.2
clip_ratio_high=0.28
PROJECT_NAME="rl-freeform"

MODEL_NAME="Qwen3-8B"
MODEL_PATH="/fast/nchandak/models/$MODEL_NAME"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=10240
TRAINER_EXPERIMENT_NAME="$MODEL_NAME-$MAX_PROMPT_LENGTH-$MAX_RESPONSE_LENGTH"

# Initialize Ray with minimal configuration
echo "Initializing Ray..."
python3 -c "
import ray
import time

try:
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        local_mode=False,
        num_cpus=32,
        num_gpus=8,
        object_store_memory=5000000000,  # 5GB
        _memory=10000000000,  # 10GB
        log_to_driver=False,
        include_dashboard=False,
        _redis_max_memory=500000000  # 500MB
    )
    print('Ray initialized successfully')
except Exception as e:
    print(f'Ray initialization failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "Failed to initialize Ray, exiting"
    exit 1
fi

# Run training with simplified configuration
taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
reward_model.enable=True \
reward_model.model.path=/fast/nchandak/models/Qwen3-4B \
reward_model.strategy=matcher \
reward_model.reward_manager=naive \
reward_model.micro_batch_size=0 \
data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/datamix/combined_non_numeric_all_train.jsonl \
data.val_files=/fast/nchandak/forecasting/datasets/verl/freeform/datamix/combined_non_numeric_all_validation.jsonl \
data.train_batch_size=128 \
data.val_batch_size=64 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=5e-6 \
actor_rollout_ref.actor.optim.warmup_style=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.01 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.critic_warmup=0 \
trainer.logger=['console','wandb'] \
trainer.n_gpus_per_node=$NUM_GPUS \
trainer.nnodes=1 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.save_freq=200 \
trainer.test_freq=30 \
trainer.total_epochs=5 \
trainer.default_local_dir="/fast/nchandak/forecasting/training/verl/checkpoints/${PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}" \
reward_manager='prime' 