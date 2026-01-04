#!/bin/bash
#
# Simplified wrapper script for launching training via Python launcher.
# All configuration is now managed through launch_training.py
#
# Usage: ./run_training.sh [python script args...]
# Example: ./run_training.sh --lr 1e-5 --total_epochs 10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Forward all arguments to the Python launcher
python3 "${SCRIPT_DIR}/launch_training.py" "$@"

