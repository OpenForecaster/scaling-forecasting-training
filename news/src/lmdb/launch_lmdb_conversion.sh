#!/bin/bash

# Activate appropriate conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Parse arguments
JSON_DIR=$1
OUTPUT_DIR=$2
shift 2
EXTRA_ARGS="$@"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the LMDB conversion script with the provided arguments
python /is/cluster/sgoel/forecasting-rl/news/to_lmdb.py "$JSON_DIR" "$OUTPUT_DIR" $EXTRA_ARGS

conda deactivate