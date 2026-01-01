#!/bin/bash

# Activate the conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Change to the project directory
cd /is/cluster/sgoel/forecasting-rl/

# Parse arguments
MODEL_PATH=$1
USE_OPENROUTER=$2
OPENROUTER_MODEL=$3
ARTICLE_PATH=$4
OUTPUT_PATH=$5
MAX_TOKENS=$6
TEMPERATURE=$7
BATCH_SIZE=$8
REGENERATE=$9

# Prepare command
CMD="python /is/cluster/sgoel/forecasting-rl/qgen/from_article.py"

# Add model path if provided (handle 'None' string)
if [ -n "$MODEL_PATH" ] && [ "$MODEL_PATH" != "None" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
fi

# Add OpenRouter flag if enabled
if [ "$USE_OPENROUTER" -eq 1 ]; then
    CMD="$CMD --use_openrouter --openrouter_model $OPENROUTER_MODEL"
fi

# Add remaining arguments
CMD="$CMD --article_path $ARTICLE_PATH --output_path $OUTPUT_PATH --max_tokens $MAX_TOKENS --temperature $TEMPERATURE --batch_size $BATCH_SIZE"

# Add regenerate flag if enabled
if [ "$REGENERATE" -eq 1 ]; then
    CMD="$CMD --regenerate"
fi

# Execute the command
echo "Executing: $CMD"
eval $CMD

# Deactivate conda environment
conda deactivate