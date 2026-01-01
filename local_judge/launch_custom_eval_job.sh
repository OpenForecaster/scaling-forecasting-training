#!/bin/bash

source /home/nchandak/miniforge3/bin/activate minir1
module load cuda/12.1

python llm_judge.py --input_file $1 

conda deactivate