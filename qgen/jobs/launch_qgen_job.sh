#!/bin/bash

# Usage: launch_qgen_job.sh <article_path> <output_dir> [additional_args...]
# Example: launch_qgen_job.sh articles.jsonl ./outputs --cutoff_date 2025-05-01

# Initialize conda if available, otherwise skip
if command -v conda &> /dev/null; then
    conda deactivate
fi

cd /home/nchandak/forecasting/
source forecast/bin/activate
module load cuda/12.1

cd /home/nchandak/forecasting/qgen

# Run the new pipeline (quality defaults always enabled: freeq, leakage check, best selection, validation, date updates)
# $1 = article_path (required)
# $2 = output_dir (required)
# $3+ = additional arguments (optional)
python3 run_pipeline.py \
    --article_path "$1" \
    --output_dir "$2" \
    --num_q_per_article 3 \
    --use_openrouter \
    ${@:3}

# Clean up environment if conda is available
if command -v conda &> /dev/null; then
    conda deactivate
fi