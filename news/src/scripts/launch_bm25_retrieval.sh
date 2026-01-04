#!/bin/bash

# Activate appropriate conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Run the BM25 retrieval script with the provided arguments
python /is/cluster/sgoel/forecasting-rl/news/src/bm25_jsonl.py $@

conda deactivate