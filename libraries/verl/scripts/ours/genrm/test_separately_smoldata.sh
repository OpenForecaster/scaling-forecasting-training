#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Configure Ray to properly handle GPU allocation
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_DISABLE_IMPORT_WARNING=1

# Ray configuration for GPU isolation
export RAY_memory_usage_threshold=0.95
export RAY_object_store_memory_threshold=0.95

export HYDRA_FULL_ERROR=1

# Additional network configuration to avoid SSL/proxy issues
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export SSL_VERIFY=false
export PYTHONHTTPSVERIFY=0
export PYTHONIOENCODING=utf-8

# Disable proxy for localhost connections
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# Configure Wandb - use offline mode to avoid network issues
echo "Configuring Wandb in offline mode to avoid network connectivity issues"
export WANDB_MODE=offline
export WANDB_DIR=/tmp/wandb
export WANDB_INIT_TIMEOUT=300
export WANDB_SILENT=true
export WANDB_CONSOLE=off
LOGGER_CONFIG="['console','wandb']"

# Create wandb directory
mkdir -p /tmp/wandb

# Configure Ray to allow GPU sharing (not isolation)
# export RAY_GPU_MEMORY_FRACTION=0.8
# export RAY_GPU_MEMORY_ALLOW_GROWTH=true

# /fast/rolmedo/models/qwen2.5-7b-it 
# 'qwen2.5-7b-datamix-train-brier'


# actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
# actor_rollout_ref.rollout.disable_chunked_prefill=True \
# +actor_rollout_ref.actor.fsdp_config.grad_offload=True \

# reward_model.model.path=/fast/nchandak/models/general-verifier \
# actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \

# algorithm.norm_adv_by_std_in_grpo=False \
# data.max_prompt_length=1024 \
# data.max_response_length=7168 \
# actor_rollout_ref.rollout.enable_chunked_prefill=False \
# actor_rollout_ref.rollout.max_num_batched_tokens=16384 \

# Clean up any existing Ray or vLLM processes
echo "Cleaning up any existing Ray or vLLM processes..."
pkill -f "ray::" 2>/dev/null || true
pkill -f "vllm" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

# Start vLLM server for Qwen3-4B model on GPU 7
echo "Starting vLLM server for Qwen3-4B model on GPU 7..."
CUDA_VISIBLE_DEVICES=7 vllm serve /fast/nchandak/models/Qwen3-4B --served-model-name qwen3-4b-non-think --port 30000 --host 0.0.0.0 --gpu-memory-utilization 0.8 &
VLLM_PID=$!

# Wait a moment for the server to start
sleep 15

# Check if the vLLM server is running properly
echo "Checking if vLLM server is running..."
for i in {1..30}; do
    if curl -s --noproxy "*" http://127.0.0.1:30000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is running successfully!"
        break
    else
        echo "Waiting for vLLM server to start... (attempt $i/30)"
        sleep 5
    fi
done

# Final check
if ! curl -s --noproxy "*" http://127.0.0.1:30000/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start properly!"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Test the chat completions endpoint specifically
echo "Testing chat completions endpoint..."
test_response=$(curl -s --noproxy "*" -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b-non-think",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' 2>&1)

if echo "$test_response" | grep -q "Access Denied\|ERR_ACCESS_DENIED\|squid"; then
    echo "ERROR: Proxy/firewall is blocking access to vLLM server!"
    echo "Please check your network configuration and proxy settings."
    echo "Response: $test_response"
    kill $VLLM_PID 2>/dev/null
    exit 1
elif echo "$test_response" | grep -q "choices\|error"; then
    echo "Chat completions endpoint is working!"
else
    echo "WARNING: Unexpected response from chat completions endpoint"
    echo "Response: $test_response"
fi

# Wait for vLLM server to fully stabilize before starting training
echo "Waiting for vLLM server to stabilize..."
sleep 10

# # Initialize Ray to check if it works properly
# echo "Testing Ray initialization..."
# ray start --head --num-cpus=8 --num-gpus=7 --include-dashboard=false || {
#     echo "ERROR: Ray failed to start!"
#     exit 1
# }

# # Stop Ray after test
# ray stop

# Get the number of available GPUs for training
NUM_GPUS=7 # Use GPUs 0-6 for training
echo "Number of available GPUs for training: $NUM_GPUS"

# Set CUDA_VISIBLE_DEVICES to use GPUs 0-6 for training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

clip_ratio_low=0.2
clip_ratio_high=0.28
PROJECT_NAME="genrm-data22k"

MODEL_NAME="Qwen3-4B-Instruct-2507"
MODEL_NAME="Qwen3-4B-Base"
MODEL_NAME="Qwen3-4B" # -Thinking-2507"
MODEL_PATH="/fast/nchandak/models/$MODEL_NAME"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
TRAINER_EXPERIMENT_NAME="$MODEL_NAME-$MAX_PROMPT_LENGTH-$MAX_RESPONSE_LENGTH"

taskset -c 0-31 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
reward_model.reward_manager=batch \
custom_reward_function.path=scripts/ours/genrm/brier_reward_function.py \
custom_reward_function.name=compute_score_batch \
data.train_files=/fast/nchandak/forecasting/datasets/verl/freeform/data22k-withbinary/combined_all_questions_with_binary_train.jsonl \
data.val_files=/fast/nchandak/forecasting/datasets/verl/freeform/data22k-withbinary/combined_all_questions_validation.jsonl \
data.train_batch_size=256 \
data.val_batch_size=256 \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.truncation='error' \
data.filter_overlong_prompts=True \
actor_rollout_ref.model.path=$MODEL_PATH \
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
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.rollout.n=7 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
trainer.critic_warmup=0 \
trainer.logger=$LOGGER_CONFIG \
trainer.n_gpus_per_node=$NUM_GPUS \
trainer.nnodes=1 \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$TRAINER_EXPERIMENT_NAME \
trainer.save_freq=50 \
trainer.test_freq=20 \
trainer.total_epochs=8 \
trainer.default_local_dir="/fast/nchandak/forecasting/training/verl/checkpoints/${PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}" \
+reward_manager='prime'

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    
    # Kill Ray processes
    pkill -f "ray::" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    
    # Kill vLLM server
    if [[ ! -z "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    
    echo "Cleanup completed."
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Cleanup: Kill the vLLM server when training is done
echo "Training completed. Stopping vLLM server..."
cleanup