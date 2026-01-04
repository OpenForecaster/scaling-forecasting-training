#!/bin/bash

# Activate appropriate conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Run the article selection script with the provided arguments
python /is/cluster/sgoel/forecasting-rl/qgen/select_articles.py $@

conda deactivate