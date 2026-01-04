#!/bin/bash

# Single-Node Multi-GPU GRPO LoRA Training Launch Script
# This script launches the training using Accelerate for multi-GPU support on a single node

# Set default values
MODEL_PATH="/fast/nchandak/models/Qwen3-4B-Base"
DATASET_NAME="HuggingFaceH4/OpenR1-Math-220k-default-verified"
OUTPUT_DIR="/lustre/scratch/nchandak/forecasting/training/lora_without_regret/qwen3-4b-base"
RUN_NAME="grpo-lora-qwen3-4b-base-multigpu"
LEARNING_RATE=1e-6
LORA_R=16
LORA_TARGET_MODULES="all-linear"

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
        --bf16)
            BF16_FLAG="--bf16"
            shift
            ;;
        --fp16)
            FP16_FLAG="--fp16"
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

# Launch training with Accelerate (single-node multi-GPU)
accelerate launch \
    --config_file accelerate_config.yaml \
    lora_without_regret.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --lora_r "$LORA_R" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_prompt_length 1024 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --seed 42 \
    $BF16_FLAG \
    $FP16_FLAG

echo "Training completed!"
