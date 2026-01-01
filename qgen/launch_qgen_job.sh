#!/bin/bash

# Initialize conda if available, otherwise skip
if command -v conda &> /dev/null; then
    conda deactivate
fi

cd /home/nchandak/forecasting/
source forecast/bin/activate
module load cuda/12.1

cd /home/nchandak/forecasting/qgen
# bash scripts/iterate_datamix.sh
python3 from_article.py --article_path $1 --num_q 3 --freeq --check_leakage --choose_best --validate --use_openrouter
# bash scripts/check_halawi_train.sh $1 $2

# Clean up environment if conda is available
if command -v conda &> /dev/null; then
    conda deactivate
fi