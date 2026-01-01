#!/bin/bash

# VENV_PATH="/home/nchandak/miniforge3/envs/trainingtt/"

# source "$VENV_PATH/bin/activate"

# Initialize conda if available, otherwise skip
if command -v conda &> /dev/null; then
    conda deactivate
fi


# # Set Python path
# if [ -z "${PYTHONPATH:-}" ]; then
#     export PYTHONPATH="/lustre/home/nchandak/forecasting/libraries/verl"
# else
#     export PYTHONPATH="${PYTHONPATH}:/lustre/home/nchandak/forecasting/libraries/verl"
# fi

cd /home/nchandak/forecasting/
source forecast/bin/activate
module load cuda/12.1

cd /home/nchandak/forecasting/custom_eval_scripts

# bash scripts/iterate_datamix.sh
# bash scripts/ours/trial/llama_forecasting.sh
# bash scripts/ours/trial/test_forecasting.sh
# bash scripts/ours/trial/test_retrieval.sh
# bash scripts/ours/forbes24/test_past.sh
# bash scripts/ours/listoutcomes/test_20k.sh
# bash scripts/ours/trial/test_smoldata.sh
# bash scripts/ours/trial/run_thinking_model.sh
# bash scripts/check_halawi_train.sh $1 $2

# Add error handling for the script execution
# echo "Starting training script..."
# if bash scripts/ours/trial/run_thinking_model.sh; then
#     echo "Training completed successfully"
# else
#     echo "Training failed with exit code $?"
#     exit 1
# fi

# python eval_freeform.py --num_generations 8192 --model_dir /fast/nchandak/models/Qwen3-8B
# python eval_freeform.py --num_generations 8192 --model_dir /fast/nchandak/models/Qwen3-8B
# python eval_withretrieval.py --num_generations 1024 --model_dir /fast/nchandak/models/Qwen3-8B --questions_file /fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-new-30_207_free_3_cleaned.jsonl
# python eval_withretrieval.py --num_generations 1024 --model_dir /fast/nchandak/forecasting/training/verl/checkpoints/distill-grok-3-mini/useful/Qwen3-8B-8192-4096-1e-4-distill-grok-3-mini-checkpoint108 --questions_file /fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-new-30_207_free_3_cleaned.jsonl

# python eval_retrieval.py --num_generations 1024 --model_dir /fast/nchandak/forecasting/training/verl/checkpoints/distill-grok-3-mini/useful/Qwen3-8B-8192-4096-1e-4-distill-grok-3-mini-checkpoint108 --questions_file /fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_validation-retrieval_207_30.jsonl --num_articles 10


# python eval_retrieval.py --num_generations 1024 --model_dir $1 --questions_file /fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_validation-retrieval_207_30.jsonl --num_articles $2

# python batch_freeform_evals.py  --mode withretrieval --questions_file fivenews --input_dir /fast/nchandak/forecasting/training/verl/checkpoints/rl-retrieval-data70k/Qwen3-8B-sft-4096-6144-randomKarticles-acc-4e-6-resume-kl0.01

python batch_freeform_evals.py --input_dir /lustre/scratch/nchandak/forecasting/training/verl/checkpoints/rl-retrievalfixed-data70k/llama-3.2-3b-it-4096-5e-6-kl0.005 --questions_file fivenews

# Clean up environment if conda is available
if command -v conda &> /dev/null; then
    conda deactivate
fi