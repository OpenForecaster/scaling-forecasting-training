#!/bin/bash

# Activate appropriate conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Run the tokenization script with the provided arguments
python /is/cluster/sgoel/forecasting-rl/news/src/tokenize_for_rag.py $@

conda deactivate