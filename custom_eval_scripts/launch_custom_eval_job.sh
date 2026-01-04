#!/bin/bash

# HTCondor job launcher script for custom evaluations.
# Activates conda environment, loads CUDA, and dispatches to appropriate eval script.
# Supports multiple task types: forecasting, mcq_forecasting, mmlu-pro, math, freeform, simpleqa, retrieval.
# Called by jobs_eval.py when submitting HTCondor jobs.

export WANDB_API_KEY=""
# export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"

source /home/nchandak/miniforge3/bin/activate minir1
module load cuda/12.1

# Get the task type from the first argument
TASK=${1:-"forecasting"}  # Default to forecasting if not specified

if [ "$TASK" = "forecasting" ]; then
    python eval_forecasting_vllm.py --base_save_dir $2 --model_dir $3 --model $4 --max_new_tokens $5 --data_split $6 --num_generations $7 --data $8 
elif [ "$TASK" = "mcq_forecasting" ]; then
    python eval_forecasting_mcq.py --base_save_dir $2 --model_dir $3 --model $4 --max_new_tokens $5 --data_split $6 --num_generations $7 --data $8
elif [ "$TASK" = "mmlu-pro" ]; then
    python eval_mmlu_pro.py --base_save_dir $2 --model_dir $3 --model $4 --max_new_tokens $5 --data_split $6 --num_generations $7 --data $8
elif [ "$TASK" = "math" ]; then
    python eval_math.py --base_save_dir $2 --model_dir $3 --model $4 --max_new_tokens $5 --data_split $6 --num_generations $7 --data $8
elif [ "$TASK" = "freeform" ]; then
    python eval_freeform.py --base_save_dir $2 --model_dir $3 --model $4 --max_new_tokens $5 --data_split $6 --num_generations $7 --questions_file $8
elif [ "$TASK" = "simpleqa" ]; then
    python eval_simpleqa.py --model_dir $3
elif [ "$TASK" = "retrieval" ]; then
    python eval_withretrieval.py --model_dir $3 --num_generations $7 --questions_file $8 --num_articles $9
else
    echo "Unknown task: $TASK"
    exit 1
fi

conda deactivate