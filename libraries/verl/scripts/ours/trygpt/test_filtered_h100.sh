#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Configure Ray to properly handle GPU allocation
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1

export HYDRA_FULL_ERROR=1


# B200 GPU specific fixes
export VLLM_DISABLE_CUDAGRAPH=1
export VLLM_USE_TRITON_FLASH_ATTN=0
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_ENFORCE_EAGER=1

# Additional B200 memory fixes
export VLLM_DISABLE_CUSTOM_KERNELS=1
export VLLM_USE_FLASHINFER=0
export VLLM_DISABLE_CUSTOM_ALLREDUCE=1
export TORCH_CUDNN_V8_API_ENABLED=0

# Memory management for B200
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# FlashAttention compatibility
# export FLASH_ATTENTION_FORCE_BUILD=0
# export FLASH_ATTENTION_SKIP_CUDA_BUILD=1

# Fix GPT-OSS model compatibility issues
# export VLLM_DISABLE_HARMONY=1  # Keep harmony enabled for GPT-OSS
export VLLM_DISABLE_MFU=1
export VLLM_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

# Ensure harmony encoding can find its files
export HARMONY_TOKENIZER_PATH=/fast/nchandak/models/gpt-oss-20b-bf16
# export PYTHONPATH=/lustre/home/nchandak/forecasting/gptoss/lib/python3.12/site-packages:$PYTHONPATH

# Additional environment variables for harmony encoding
export HARMONY_CACHE_DIR=/tmp/harmony_cache
export HARMONY_DATA_DIR=/fast/nchandak/models/gpt-oss-20b-bf16
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache

# Create cache directories
mkdir -p $HARMONY_CACHE_DIR
mkdir -p $TIKTOKEN_CACHE_DIR

# Fix vLLM torch compilation cache issues
# Disable torch compilation since we don't have a suitable local filesystem for cache
# export VLLM_TORCH_COMPILE_LEVEL=0
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Commented out - causes FlashAttention 3 assertion error

# +actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=None \

# Disable Hugging Face Hub access to prevent safetensors retrieval errors
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# actor_rollout_ref.actor.use_torch_compile=False \
# actor_rollout_ref.ref.use_torch_compile=False \
# +reward_model.engine_kwargs.vllm.enforce_eager=True \

# Disable TensorRT-LLM to prevent compatibility issues with GPT-OSS
# export VLLM_USE_TRTLLM=0
# export VLLM_ATTENTION_BACKEND=XFORMERS  # Removed due to FlashAttention version conflict

# Configure Ray to allow GPU sharing (not isolation)
# export RAY_GPU_MEMORY_FRACTION=0.8
# export RAY_GPU_MEMORY_ALLOW_GROWTH=true

# /fast/rolmedo/models/qwen2.5-7b-it 
# 'qwen2.5-7b-datamix-train-brier'


# actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
# actor_rollout_ref.rollout.disable_chunked_prefill=True \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=True \

# +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
# +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \
# +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \
# +reward_model.engine_kwargs.vllm.enforce_eager=True \
# +actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \

# reward_model.model.path=/fast/nchandak/models/general-verifier \
# actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \

# data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/forbes24/filtered_train_10k.jsonl \
# data.val_files=/fast/nchandak/forecasting/datasets/verl/freeform/data22k-withbinary/combined_all_questions_validation.jsonl \

# data.train_files=/fast/nchandak/forecasting/misc/gsm8k/train.parquet \
# data.val_files=/fast/nchandak/forecasting/misc/gsm8k/test.parquet \

# algorithm.norm_adv_by_std_in_grpo=False \
# data.max_prompt_length=1024 \
# data.max_response_length=7168 \
# actor_rollout_ref.rollout.enable_chunked_prefill=False \
# actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
# Get the number of available GPUs
NUM_GPUS=8 # Use only first 4 GPUs (0,1,2,3) for training
echo "Number of available GPUs: $NUM_GPUS"

# Set CUDA_VISIBLE_DEVICES to use GPUs 0-3 for training and 7 for verifier
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

clip_ratio_low=0.2
clip_ratio_high=0.28
PROJECT_NAME="rl-forbes24"

MODEL_NAME="gpt-oss-20b-bf16" # -Thinking-2507"
# MODEL_NAME="Qwen3-0.6B" # -Thinking-2507"
MODEL_PATH="/fast/nchandak/models/$MODEL_NAME"

# Verify model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path $MODEL_PATH does not exist!"
    exit 1
fi
echo "Using model path: $MODEL_PATH"

# Test harmony encoding availability
echo "Testing harmony encoding..."
python3 -c "
try:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print('✓ Harmony encoding loaded successfully')
except Exception as e:
    print(f'✗ Harmony encoding failed: {e}')
    exit(1)
"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096 # 6144
TRAINER_EXPERIMENT_NAME="filtered-$MODEL_NAME-$MAX_PROMPT_LENGTH-$MAX_RESPONSE_LENGTH"

taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
reward_model.enable=True \
reward_model.model.path=/fast/nchandak/models/Qwen3-4B \
reward_model.strategy=matcher \
reward_model.reward_manager=naive \
+reward_model.engine_kwargs.vllm.enforce_eager=True \
reward_model.micro_batch_size=0 \
data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/forbes24/filtered_train_10k.jsonl \
data.val_files=/fast/nchandak/forecasting/datasets/verl/freeform/data22k-withbinary/combined_all_questions_validation.jsonl \
data.train_batch_size=64 \
data.val_batch_size=64 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
data.shuffle=False \
data.filter_overlong_prompts=True \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.actor.optim.lr=5e-6 \
actor_rollout_ref.actor.optim.warmup_style=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.01 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
+actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
+actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=True \
+actor_rollout_ref.rollout.engine_kwargs.vllm.trust_remote_code=True \
+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=8192 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
trainer.logger='["console","wandb"]' \
++trainer.val_before_train=False \
trainer.n_gpus_per_node=$NUM_GPUS \
trainer.nnodes=1 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.save_freq=500 \
trainer.test_freq=30 \
trainer.total_epochs=10 \
trainer.default_local_dir="/fast/nchandak/forecasting/training/verl/checkpoints/${PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}" \
+reward_manager='prime'