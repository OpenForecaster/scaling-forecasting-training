#!/bin/bash

# Example script for running Paleka CCFLMF benchmark evaluation.
# Demonstrates how to run eval_paleka.py with proper environment setup.
# Modify model_dir, data_dir, and output_dir according to your setup.

# Activate the environment
cd /home/nchandak/forecasting/
source forecast/bin/activate
module load cuda/12.1

cd /home/nchandak/forecasting/custom_eval_scripts

# Set your model directory and other parameters
MODEL_DIR="/fast/nchandak/models/Qwen3-1.7B"
DATA_DIR="/home/nchandak/forecasting/custom_eval_scripts/tuples_2028"
OUTPUT_DIR="/fast/nchandak/forecasting/evals/paleka"
MAX_TOKENS=16384
NUM_GENERATIONS=3

# Run the evaluation
python eval_paleka.py \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --base_save_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_TOKENS" \
    --num_generations "$NUM_GENERATIONS"

echo "Evaluation complete! Check $OUTPUT_DIR for results." 