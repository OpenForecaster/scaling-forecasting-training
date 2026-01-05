# Training Launcher

This directory contains a refactored training launcher that makes it easy to configure and run VERL PPO training jobs. The current configuration only works on GPUs with 80+ GB memory, as we load another LLM-as-a-judge on the same GPU where training occurs, following the implementation of [General Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner/tree/main).

## Files

- **`launch_training.py`**: Main Python launcher with all configuration defaults and command-line argument parsing
- **`run_training.sh`**: Simple shell wrapper that calls the Python launcher
- **`run_8b.sh.legacy`**: Original shell script (kept for reference)

## Usage


### How to run with custom params 

Please first go through this whole README to understand the args some of which may be specific to your setup/config so provide those accordingly (like num gpus, model path, etc.).

### Command Preview
To see exactly what command will be executed:
```bash
./launch_training.py --dry_run
```

Override any parameter via command-line arguments:

```bash
./launch_training.py \
    --lr 1e-5 \
    --total_epochs 10 \
    --model_name "Qwen3-8B" \
    --max_response_length 8192
```

### Common Configuration Options

#### Hardware Configuration
```bash
--num_gpus 8                         # Number of GPUs per node
--nnodes 1                           # Number of nodes
--taskset_cpus "0-31"                # CPU cores to use
```

#### Model Configuration
```bash
--model_name "Qwen3-8B"              # Model name
--model_path "/path/to/model"        # Custom model path
--reward_model_path "/path/to/rm"    # LLM-as-a-judge path (for checking model response)
```

#### Data Configuration
```bash
--train_files "/path/to/train.jsonl" # Training data
--val_files "/path/to/val.jsonl"     # Validation data
--train_batch_size 256               # Training batch size
--val_batch_size 512                 # Validation batch size
```

#### Experiment Tracking
```bash
--project_name "my-project"          # Project name for logging
--experiment_name "my-experiment"    # Experiment name (auto-generated if not provided)
--logger "console,wandb"             # Loggers to use
```


#### Training Hyperparameters
```bash
--lr 5e-6                            # Learning rate
--kl_coeff 0.005                     # KL coefficient
--clip_ratio_low 0.2                 # Lower clip ratio
--clip_ratio_high 0.28               # Higher clip ratio
--total_epochs 7                     # Total training epochs
```

#### Sequence Lengths
```bash
--max_prompt_length 4096             # Maximum prompt length
--max_response_length 4096           # Maximum response length
```

### View All Options

See all available options and their defaults:

```bash
./launch_training.py --help
```

### Dry Run

Preview the command without executing it:

```bash
./launch_training.py --dry_run
```

## Examples

### Example 1: Quick Test with Smaller Model
```bash
./launch_training.py \
    --model_name "Qwen3-4B" \
    --total_epochs 3 \
    --train_batch_size 128
```

### Example 2: Training with Custom Settings
```bash
./launch_training.py \
    --model_name "Qwen3-8B" \
    --lr 1e-5 \
    --kl_coeff 0.01 \
    --max_response_length 8192 \
    --total_epochs 10 \
    --project_name "forecasting-rl" \
    --experiment_name "qwen3-8b-openforesight"
```


### Example 5: Ablation Study
```bash
# Test different learning rates
./launch_training.py --lr 1e-6 --project_name "ablation-lr" --experiment_name "lr-1e-6"
./launch_training.py --lr 5e-6 --project_name "ablation-lr" --experiment_name "lr-5e-6"
./launch_training.py --lr 1e-5 --project_name "ablation-lr" --experiment_name "lr-1e-5"
```

## Environment Variables

The launcher automatically sets up the following environment variables:
- CUDA configuration (CUDA_LAUNCH_BLOCKING, TORCH_USE_CUDA_DSA, etc.)
- Ray configuration (RAY_DEDUP_LOGS, RAY_IGNORE_UNHANDLED_ERRORS, etc.)
- VLLM configuration (VLLM_ATTENTION_BACKEND)
