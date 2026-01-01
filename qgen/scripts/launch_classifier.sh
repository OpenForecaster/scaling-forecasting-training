#!/bin/bash

# Load WANDB API key from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/wandb_key.txt" ]; then
    export WANDB_API_KEY=$(cat "$SCRIPT_DIR/wandb_key.txt" | tr -d '[:space:]')
else
    echo "Error: wandb_key.txt not found in $SCRIPT_DIR"
    exit 1
fi

source /home/sgoel/miniforge3/bin/activate lmfact
module load cuda/12.1
cd /is/cluster/sgoel/forecasting-rl/qgen/

# Run the classifier with accelerate
accelerate launch classifier.py \
    --data_path "$1" \
    --model_name "$2" \
    --output_dir "$3" \
    --wandb_project "$4" \
    --wandb_run_name "$5" \
    --train_ratio "$6" \
    --test_ratio "$7" \
    --num_train_epochs "$8" \
    --gradacc_steps "$9" \
    ${10} \
    ${11}

conda deactivate