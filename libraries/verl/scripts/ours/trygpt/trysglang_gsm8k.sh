#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Configure Ray to properly handle GPU allocation
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

export HYDRA_FULL_ERROR=1

# Disable FlashInfer compilation to avoid nvcc issues

# export FLASHINFER_DISABLE_JIT=1
# export FLASHINFER_DISABLE_CUSTOM_KERNELS=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_FLASHINFER=0
# export VLLM_DISABLE_CUSTOM_ALLREDUCE=1

# export FLASHINFER_DISABLE_JIT=1
# export FLASHINFER_DISABLE_CUSTOM_KERNELS=1
# export VLLM_ATTENTION_BACKEND=XFORMERS


# export VLLM_USE_FLASHINFER=0
# +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \

# export VLLM_TORCH_COMPILE_LEVEL=0
# Configure Ray to allow GPU sharing (not isolation)
# export RAY_GPU_MEMORY_FRACTION=0.8
# export RAY_GPU_MEMORY_ALLOW_GROWTH=true

# # Enable CUDA graphs for better performance
# export VLLM_USE_CUDAGRAPH=1
# export VLLM_USE_TRITON_FLASH_ATTN=1
# export VLLM_CUDAGRAPH_WARMUP_STEPS=3
# export VLLM_ENABLE_CUDAGRAPH=1

# Fix GPT-OSS model compatibility issues
# export VLLM_DISABLE_HARMONY=1
# export VLLM_DISABLE_MFU=1
# export VLLM_TRUST_REMOTE_CODE=1
# export TOKENIZERS_PARALLELISM=false

# /fast/rolmedo/models/qwen2.5-7b-it 
# 'qwen2.5-7b-datamix-train-brier'
# +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=None \


# +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_harmony=True \
# +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mfu=True \
# +actor_rollout_ref.rollout.engine_kwargs.vllm.trust_remote_code=True \

# actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
# actor_rollout_ref.rollout.disable_chunked_prefill=True \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=True \

# reward_model.model.path=/fast/nchandak/models/general-verifier \
# actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \

# +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \

# algorithm.norm_adv_by_std_in_grpo=False \
# data.max_prompt_length=1024 \
# data.max_response_length=7168 \
# actor_rollout_ref.rollout.enable_chunked_prefill=False \
# actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
# Get the number of available GPUs
NUM_GPUS=8 # Use only first 4 GPUs (0,1,2,3) for training
echo "Number of available GPUs: $NUM_GPUS"

# +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
# actor_rollout_ref.rollout.enable_chunked_prefill=True \


# Set CUDA_VISIBLE_DEVICES to use GPUs 0-3 for training and 7 for verifier
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clip_ratio_low=0.2
clip_ratio_high=0.28
PROJECT_NAME="rl-forbes24"

MODEL_NAME="Qwen3-4B" # -Thinking-2507"
MODEL_PATH="/fast/nchandak/models/$MODEL_NAME"

model_dir=/fast/nchandak/models/gpt-oss-20b-bf16

MODEL_PATH=${model_dir}
MODEL_NAME="gpt-oss-20b-bf16"
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096 # 6144
TRAINER_EXPERIMENT_NAME="gsm8k-$MODEL_NAME-$MAX_PROMPT_LENGTH-$MAX_RESPONSE_LENGTH"




taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=/fast/nchandak/forecasting/misc/gsm8k/train.parquet \
data.val_files=/fast/nchandak/forecasting/misc/gsm8k/test.parquet \
data.train_batch_size=64 \
data.max_prompt_length=${MAX_PROMPT_LENGTH} \
data.max_response_length=${MAX_RESPONSE_LENGTH} \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.entropy_coeff=0.05 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.05 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.use_kl_in_reward=False \
trainer.critic_warmup=0 \
trainer.logger='["console","wandb"]' \
trainer.project_name='verl_grpo_example_gsm8k_math' \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.n_gpus_per_node=8 \
trainer.nnodes=1 \
trainer.save_freq=10000 \
trainer.test_freq=10 \
trainer.total_epochs=50 $@