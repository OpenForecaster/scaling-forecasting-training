#!/bin/bash

# Load CUDA module to make nvcc available
module load cuda/12.1 2>/dev/null || echo "Warning: Could not load CUDA module"

# Fix CUDA environment issues for job submission
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Add CUDA to PATH (critical fix for nvcc not found)
# export PATH="/usr/local/cuda/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# export CUDA_HOME="/usr/local/cuda"

# Disable JIT compilation to avoid nvcc issues
# export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
# export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Completely disable all CUDA JIT compilation
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=0
# export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Disable FlashInfer completely
# export FLASHINFER_DISABLE_JIT=1
# export VLLM_DISABLE_FLASHINFER=1

# Prevent any CUDA compilation
# export TORCH_CUDA_ARCH_LIST=""
# export MAX_JOBS=1
# export NVCC_THREADS=1
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Additional stability settings
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600

# Fix Ray environment issues
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
# export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

# Fix serialization issues
# export PYTHONPATH="${PYTHONPATH}:/lustre/home/nchandak/forecasting/libraries/verl"
export RAY_DISABLE_IMPORT_WARNING=1

export HYDRA_FULL_ERROR=1

# Additional environment variables for multi-node training
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=INFO
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Additional Ray stability fixes
export RAY_DISABLE_STRICT_VERSION_CHECK=1
export RAY_CLIENT_FORCE_CLEANUP=1
export RAY_JOB_SUBMISSION_TIMEOUT=300

# Network and timeout configurations
export RAY_DASHBOARD_GRPC_TIMEOUT=60
export RAY_DASHBOARD_HTTP_TIMEOUT=60

# Configure Ray to allow GPU sharing (not isolation)
# export RAY_GPU_MEMORY_FRACTION=0.8
# export RAY_GPU_MEMORY_ALLOW_GROWTH=true

# /fast/rolmedo/models/qwen2.5-7b-it 
# 'qwen2.5-7b-datamix-train-brier'


# actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
# actor_rollout_ref.rollout.disable_chunked_prefill=True \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
# data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/data20k-retrieval/retrieval_all_questions_train.jsonl \
# data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/data70k-retrieval/ranked_queries_train_30-5.jsonl \
# data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/data20k-retrieval/retrieval_all_questions_train.jsonl \


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
PROJECT_NAME="rl-retrieval-data70k"
LR=4e-6
# MODEL_NAME="Qwen3-4B-Instruct-2507" # -Thinking-2507"
MODEL_NAME="Qwen3-4B"
MODEL_PATH="/fast/nchandak/forecasting/training/verl/checkpoints/rl-retrieval-data70k/goodcheckpoints/Qwen3-4B"
MAX_PROMPT_LENGTH=4096 # 12288
MAX_RESPONSE_LENGTH=6144 # 8192
TRAINER_EXPERIMENT_NAME="$MODEL_NAME-sft-$MAX_PROMPT_LENGTH-$MAX_RESPONSE_LENGTH-5articles-acc-$LR-resume"

# Check Ray cluster status before starting training
echo "Checking Ray cluster status..."
ray status --address="172.22.8.5:6379" || {
    echo "Ray cluster not available. Please start Ray cluster first."
    echo "Run: ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379"
    exit 1
}

# Wait a moment for cluster to stabilize
sleep 5

echo "Starting training directly on Ray cluster..."

# For multi-node training, don't use RAY_ADDRESS as it forces client mode
# Instead, let the training script connect to the local Ray cluster directly
unset RAY_ADDRESS

taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
reward_model.enable=True \
reward_model.model.path=/fast/nchandak/models/Qwen3-4B \
reward_model.strategy=matcher \
reward_model.reward_manager=naive \
+reward_model.add_correctness=True \
reward_model.micro_batch_size=0 \
data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/data20k-retrieval/retrieval_all_questions_train.jsonl \
data.val_files=/fast/nchandak/forecasting/datasets/verl/freeform/data70k-retrieval/combined_non_numeric_all_validation-5.jsonl \
data.train_batch_size=256 \
data.val_batch_size=512 \
data.shuffle=False \
data.filter_overlong_prompts=True \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.warmup_style=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0001 \
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
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
trainer.nnodes=2 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.save_freq=100 \
trainer.test_freq=80 \
trainer.total_epochs=6 \
trainer.default_local_dir="/fast/nchandak/forecasting/training/verl/checkpoints/${PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}" \
+reward_manager='prime'