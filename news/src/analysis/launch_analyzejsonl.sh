#!/bin/bash

source /home/sgoel/miniforge3/bin/activate lmfact
cd /is/cluster/sgoel/forecasting-rl/news/

# Run the analysis script
python src/analysis/analyzejsonl.py \
    --input "$1" \
    --output "$2"

conda deactivate