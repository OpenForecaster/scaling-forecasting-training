#!/bin/bash

# Activate the news environment
# source /home/sgoel/miniforge3/bin/activate news


cd /home/nchandak/forecasting/
source forecast/bin/activate
module load cuda/12.1

cd /home/nchandak/forecasting/news

# Run the commoncrawl script with the provided arguments
cd news-please
python3 -m newsplease.examples.commoncrawl $1 $2 $3 $4

# Deactivate the environment
conda deactivate