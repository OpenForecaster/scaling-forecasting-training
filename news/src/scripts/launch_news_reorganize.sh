#!/bin/bash

# Activate appropriate conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Run the analyze script with the provided arguments
# $1: news_dir (optional, defaults to script's internal value)
python /is/cluster/sgoel/forecasting-rl/news/src/reorganize_jsons_month.py $1

conda deactivate