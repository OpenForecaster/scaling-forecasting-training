#!/bin/bash

# Single-Node Multi-GPU GRPO Training Launch Script for GPT-OSS
# This script launches the training using Accelerate for multi-GPU support on a single node

# Set default values
# MODEL_PATH="unsloth/gpt-oss-20b"
MODEL_PATH="/fast/nchandak/models/gpt-oss-20b-bf16"
DATASET_NAME="HuggingFaceH4/OpenR1-Math-220k-default-verified"
OUTPUT_DIR="/lustre/scratch/nchandak/forecasting/training/gptoss_grpo/gpt-oss-20b"
RUN_NAME="gptoss-grpo-20b-multigpu"
LEARNING_RATE=5e-5
LORA_R=16
LORA_TARGET_MODULES="all-linear"
MAX_SAMPLES=1000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_target_modules)
            LORA_TARGET_MODULES="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --bf16)
            BF16_FLAG="--bf16"
            shift
            ;;
        --fp16)
            FP16_FLAG="--fp16"
            shift
            ;;
        --load_in_4bit)
            LOAD_4BIT_FLAG="--load_in_4bit"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set environment variables for NCCL stability and memory optimization
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch training with Accelerate (single-node multi-GPU)
accelerate launch \
    --config_file accelerate_config.yaml \
    multigpu_gptoss.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --lora_r "$LORA_R" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --max_samples "$MAX_SAMPLES" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --seed 42 \
    --use_unsloth \
    $LOAD_4BIT_FLAG \
    $BF16_FLAG \
    $FP16_FLAG

echo "Training completed!"
