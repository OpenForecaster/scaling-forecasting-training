#!/bin/bash

# Test vLLM inference with GPT-OSS 20B BF16

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

# # Initialize conda if available, otherwise skip
# if command -v conda &> /dev/null; then
#     conda deactivate
# fi

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


# Activate environment
source /lustre/home/nchandak/forecasting/gptoss/bin/activate

# Load CUDA module
module load cuda/12.9 

# Set CUDA paths from module (don't hardcode paths)
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA is working
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"

# Check if nvcc is available, if not, set additional fallback environment variables
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found, setting additional fallback environment variables"
    export FLASHINFER_DISABLE_JIT=1
    export FLASHINFER_DISABLE_CUSTOM_KERNELS=1
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export VLLM_USE_FLASHINFER=0
    export VLLM_DISABLE_CUSTOM_ALLREDUCE=1
else
    echo "nvcc version: $(nvcc --version)"
fi

# Set additional vLLM environment variables to enable CUDA graphs
export VLLM_USE_CUDAGRAPH=1
export VLLM_USE_TRITON_FLASH_ATTN=1

# Fix GPT-OSS model compatibility issues
# export VLLM_DISABLE_HARMONY=1  # Keep harmony enabled for GPT-OSS
export VLLM_DISABLE_MFU=1
export VLLM_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false

# Ensure harmony encoding can find its files
export HARMONY_TOKENIZER_PATH=/fast/nchandak/models/gpt-oss-20b-bf16
# export PYTHONPATH=/lustre/home/nchandak/forecasting/gptoss/lib/python3.12/site-packages:$PYTHONPATH

# Additional environment variables for harmony encoding
export HARMONY_CACHE_DIR=/tmp/harmony_cache_$$
export HARMONY_DATA_DIR=/fast/nchandak/models/gpt-oss-20b-bf16
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache_$$

# Create cache directories
mkdir -p $HARMONY_CACHE_DIR
mkdir -p $TIKTOKEN_CACHE_DIR

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

cd /home/nchandak/forecasting/libraries/verl

# bash scripts/ours/trygpt/dummy.sh
# bash scripts/ours/trygpt/trysglang_gsm8k.sh
# bash scripts/ours/trygpt/test_filtered.sh
# bash scripts/ours/trygpt/run_gptoss.sh
bash scripts/ours/trygpt/polaris_try.sh
# bash scripts/ours/trygpt/runlonger_gptoss.sh
# bash scripts/ours/trygpt/nokl.sh


# Clean up environment if conda is available
if command -v conda &> /dev/null; then
    conda deactivate
fi