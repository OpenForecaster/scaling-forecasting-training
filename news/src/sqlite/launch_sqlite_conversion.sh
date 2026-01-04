#!/bin/bash

# Activate appropriate conda environment
source /home/sgoel/miniforge3/bin/activate minir1
module load cuda/12.1

# Parse arguments
JSON_DIR=$1
DB_PATH=$2
shift 2
EXTRA_ARGS="$@"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$DB_PATH")"

# Run the SQLite conversion script with the provided arguments
python /is/cluster/sgoel/forecasting-rl/news/debug_sqlite.py "$JSON_DIR" --db_path "$DB_PATH" $EXTRA_ARGS

conda deactivate