#!/bin/bash
# Wrapper script to push OpenForesight dataset to Hugging Face
# This script sets up the environment and runs the Python upload script

# Activate forecast uv environment (adjust path as needed)
# source /path/to/forecast/uv/env/bin/activate

# Load CUDA module
module load cuda/12.1

# Run the Python script
cd "$(dirname "$0")"
python push_to_hf.py "$@"

