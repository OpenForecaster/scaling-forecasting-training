#!/bin/bash

# Batch evaluation shell script for running evaluations across model checkpoints.
# Finds all checkpoint directories (global_step_*) and runs eval_freeform.py on each.
# Alternative to batch_freeform_evals.py with shell-based implementation.
# Supports skipping existing results and parallel evaluation.

# Default values
INPUT_DIR="/fast/nchandak/forecasting/training/verl/checkpoints/rl-data66k-withbinary2k/Qwen3-8B-2048-8192"
OUTPUT_DIR="./eval_results"
LOG_DIR="./eval_logs"
SKIP_EXISTING=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --input_dir DIR     Directory containing checkpoints (default: $INPUT_DIR)"
            echo "  --output_dir DIR    Directory to save evaluation results (default: $OUTPUT_DIR)"
            echo "  --log_dir DIR       Directory to save evaluation logs (default: $LOG_DIR)"
            echo "  --skip_existing     Skip checkpoints that already have results (default: true)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Starting batch evaluation..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "Skip existing: $SKIP_EXISTING"
echo ""

# Find all checkpoint directories
echo "Searching for checkpoints in $INPUT_DIR..."
CHECKPOINTS=($(find "$INPUT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoint directories found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "  - $(basename "$checkpoint")"
done
echo ""

# Process each checkpoint
for checkpoint_dir in "${CHECKPOINTS[@]}"; do
    step_name=$(basename "$checkpoint_dir")
    echo "Processing checkpoint: $step_name"
    
    # Extract step number for output naming
    step_num=$(echo "$step_name" | grep -o '[0-9]\+')
    
    # Find the actual model directory inside the checkpoint
    model_dirs=($(find "$checkpoint_dir" -maxdepth 1 -type d -name "*checkpoint*"))
    
    if [ ${#model_dirs[@]} -eq 0 ]; then
        echo "  Warning: No model directory found in $checkpoint_dir, skipping..."
        continue
    fi
    
    # Use the first model directory found
    model_dir="${model_dirs[0]}"
    echo "  Model directory: $(basename "$model_dir")"
    
    # Check if results already exist (if skip_existing is true)
    if [ "$SKIP_EXISTING" = "true" ]; then
        result_file="$OUTPUT_DIR/eval_results_step_${step_num}.json"
        if [ -f "$result_file" ]; then
            echo "  Results already exist for step $step_num, skipping..."
            continue
        fi
    fi
    
    # Prepare output and log file names
    output_file="$OUTPUT_DIR/eval_results_step_${step_num}.json"
    log_file="$LOG_DIR/eval_log_step_${step_num}.log"
    
    echo "  Running eval_freeform..."
    echo "  Output: $output_file"
    echo "  Log: $log_file"
    
    # Run eval_freeform
    start_time=$(date +%s)
    
    python eval_freeform.py \
        --model_dir "$model_dir" \
        --output_file "$output_file" \
        2>&1 | tee "$log_file"
    
    exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Completed successfully in ${duration}s"
    else
        echo "  ✗ Failed with exit code $exit_code after ${duration}s"
        echo "  Check log file: $log_file"
    fi
    
    echo ""
done

echo "Batch evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Logs saved in: $LOG_DIR"

# Print summary
echo ""
echo "Summary:"
echo "Total checkpoints found: ${#CHECKPOINTS[@]}"
echo "Results directory: $OUTPUT_DIR"
echo "Logs directory: $LOG_DIR"
