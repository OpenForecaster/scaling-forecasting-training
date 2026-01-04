#!/bin/bash

# VENV_PATH="/home/nchandak/miniforge3/envs/trainingtt/"

# source "$VENV_PATH/bin/activate"

export WANDB_API_KEY="7133f16727b5ee93dc95d4cbf35ebaa37e2310a4"

source /home/nchandak/miniforge3/bin/activate minir1
module load cuda/12.1

# python parallel_data_relabel.py --base_save_dir $1 --model_dir $2 --model $3 --batch_size $4
# accelerate launch parallel_data_relabel.py --base_save_dir $1 --model_dir $2 --model $3 --batch_size $4

accelerate launch --config_file  configs/zero3_single_gpu.yaml train_grpo_log_odds.py --config train_config_log_odds.yaml 
conda deactivate