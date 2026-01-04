#!/bin/bash

# Set error handling
set -e  # Exit on any error
set -u  # Exit on undefined variables

# VENV_PATH="/home/nchandak/miniforge3/envs/trainingtt/"

# source "$VENV_PATH/bin/activate"

# Load WANDB API key from file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/wandb_key.txt" ]; then
    export WANDB_API_KEY=$(cat "$SCRIPT_DIR/wandb_key.txt" | tr -d '[:space:]')
else
    echo "Error: wandb_key.txt not found in $SCRIPT_DIR"
    exit 1
fi

# Initialize conda if available, otherwise skip
if command -v conda &> /dev/null; then
    conda deactivate
fi

# Set up environment variables for job submission
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Fix Ray environment
export RAY_DEDUP_LOGS=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_IMPORT_WARNING=1

# Set Python path
if [ -z "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="/lustre/home/nchandak/forecasting/libraries/verl"
else
    export PYTHONPATH="${PYTHONPATH}:/lustre/home/nchandak/forecasting/libraries/verl"
fi

cd /home/nchandak/forecasting/
source forecast/bin/activate
module load cuda/12.1

# Verify CUDA is working
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

cd /home/nchandak/forecasting/libraries/verl

# bash scripts/iterate_datamix.sh
# bash scripts/ours/trial/llama_forecasting.sh
# bash scripts/ours/trial/test_forecasting.sh
# bash scripts/ours/trial/test_retrieval.sh
# bash scripts/ours/forbes24/test_past.sh
# bash scripts/ours/listoutcomes/test_20k.sh
# bash scripts/ours/trial/test_smoldata.sh
# bash scripts/ours/trial/run_thinking_model.sh
# bash scripts/check_halawi_train.sh $1 $2

# bash scripts/ours/sft/distill.sh
# bash scripts/ours/postsft_retrieval/run_1b_acc.sh
# bash scripts/ours/postsft_retrieval/run_4b.sh
# bash scripts/ours/postsft_retrieval/run_4b_binary_fresh.sh
# bash scripts/ours/postsft_retrieval/run_8b_binary.sh

# bash scripts/ours/postsft_retrieval/resume8b.sh
# bash scripts/ours/postsft_retrieval/resume_4b.sh
# bash scripts/ours/postsft_retrieval/resume8b_random.sh
# bash scripts/ours/postsft_retrieval/run_4b_future.sh
# bash scripts/ours/postsft_retrieval/run_8b_future.sh
# bash scripts/ours/postsft_retrieval/interleaved_nokl_8b.sh

# ICLR Rebuttal
bash scripts/ours/othermodels/llama.sh
# bash scripts/ours/iclr_rebuttal/gemma.sh


# bash scripts/ours/forbes24/test_with_leakage.sh
# bash scripts/ours/forbes24/test_filtered-lora.sh

# Add error handling for the script execution
# echo "Starting training script..."
# if bash scripts/ours/trial/run_thinking_model.sh; then
#     echo "Training completed successfully"
# else
#     echo "Training failed with exit code $?"
#     exit 1
# fi

# Clean up environment if conda is available
if command -v conda &> /dev/null; then
    conda deactivate
fi