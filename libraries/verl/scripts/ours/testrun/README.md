# Training Launcher

This directory contains a refactored training launcher that makes it easy to configure and run VERL PPO training jobs. The current configuration only works on GPUs with 80+ GB memory, as we load another LLM-as-a-judge on the same GPU where training occurs, following the implementation of [General Reasoner](https://github.com/TIGER-AI-Lab/General-Reasoner/tree/main).

## Files

- **`launch_training.py`**: Main Python launcher with all configuration defaults and command-line argument parsing
- **`run_training.sh`**: Simple shell wrapper that calls the Python launcher
- **`run_8b.sh.legacy`**: Original shell script (kept for reference)

## Usage

### Basic Usage

Run training with all default parameters:

```bash
./launch_training.py
# or
./run_training.sh
```

### Customizing Parameters

Override any parameter via command-line arguments:

```bash
./launch_training.py \
    --lr 1e-5 \
    --total_epochs 10 \
    --model_name "Qwen3-8B" \
    --max_response_length 8192
```

### Common Configuration Options

#### Model Configuration
```bash
--model_name "Qwen3-8B"              # Model name
--model_path "/path/to/model"        # Custom model path
--reward_model_path "/path/to/rm"    # Reward model path
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

#### Data Configuration
```bash
--train_files "/path/to/train.jsonl" # Training data
--val_files "/path/to/val.jsonl"     # Validation data
--train_batch_size 256               # Training batch size
--val_batch_size 512                 # Validation batch size
```

#### Hardware Configuration
```bash
--num_gpus 8                         # Number of GPUs per node
--nnodes 1                           # Number of nodes
--taskset_cpus "0-31"                # CPU cores to use
```

#### Experiment Tracking
```bash
--project_name "my-project"          # Project name for logging
--experiment_name "my-experiment"    # Experiment name (auto-generated if not provided)
--logger "console,wandb"             # Loggers to use
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

### Example 2: Production Training with Custom Settings
```bash
./launch_training.py \
    --model_name "Qwen3-8B" \
    --lr 1e-5 \
    --kl_coeff 0.01 \
    --max_response_length 8192 \
    --total_epochs 10 \
    --project_name "production-rl" \
    --experiment_name "qwen3-8b-high-quality"
```

### Example 3: Using Custom Data
```bash
./launch_training.py \
    --train_files "/path/to/my/train_data.jsonl" \
    --val_files "/path/to/my/val_data.jsonl" \
    --max_prompt_length 2048 \
    --max_response_length 2048
```

### Example 4: Memory Efficient Configuration
```bash
./launch_training.py \
    --model_name "Qwen3-4B" \
    --max_prompt_length 2048 \
    --max_response_length 2048 \
    --train_batch_size 128 \
    --ppo_mini_batch_size 128 \
    --gpu_memory_utilization 0.4 \
    --num_gpus 4
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

These are configured for optimal training performance and don't need to be set manually.

## Migration from Old Shell Script

If you were using the old `shuffled_8b.sh` script, simply replace:

```bash
# Old way
./shuffled_8b.sh
```

with:

```bash
# New way (with same defaults)
./launch_training.py
```

To customize parameters that were previously hardcoded in the shell script, use command-line arguments:

```bash
# Old: Edit variables in shell script
# New: Pass as arguments
./launch_training.py --lr 1e-5 --kl_coeff 0.01
```

## Advantages of the New Launcher

1. **No Script Editing**: Configure everything via command-line arguments
2. **Better Documentation**: All options documented with `--help`
3. **Type Safety**: Proper type validation for all parameters
4. **Reusability**: Easy to create multiple configurations without copying scripts
5. **Version Control**: Configuration is command history, not script modifications
6. **Scriptable**: Easy to integrate into automation pipelines

## Troubleshooting

### Command Preview
To see exactly what command will be executed:
```bash
./launch_training.py --dry_run
```

### Verify Environment
The launcher prints environment setup confirmation and a configuration summary before starting training.

### Custom Paths
If you need to use different paths than the defaults, specify them explicitly:
```bash
./launch_training.py \
    --model_path "/your/custom/path/to/model" \
    --train_files "/your/custom/data.jsonl" \
    --default_local_dir "/your/custom/checkpoint/dir"
```

