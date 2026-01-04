# Single-Node Multi-GPU GRPO LoRA Training

This setup enables distributed training across multiple GPUs on a single node using Accelerate.

## Files

- `lora_without_regret.py` - Main training script with Accelerate support
- `accelerate_config.yaml` - Single-node multi-GPU configuration
- `launch_multigpu_training.sh` - Launch script for training

## Usage

### Single Node, Multiple GPUs

```bash
# Use the launch script
./launch_multigpu_training.sh

# Or run directly with accelerate
accelerate launch --config_file accelerate_config.yaml lora_without_regret.py
```

### Custom Configuration

You can create your own accelerate config:

```bash
accelerate config
```

## Key Features

- **Multi-GPU Training**: Automatically distributes training across available GPUs on a single node
- **Mixed Precision**: Supports both FP16 and BF16 for memory efficiency
- **LoRA Integration**: Parameter-efficient fine-tuning with LoRA adapters
- **Wandb Logging**: Centralized logging (only on main process)
- **Automatic Synchronization**: Proper process synchronization for model saving

## Configuration Options

### Accelerate Config Parameters

- `num_machines`: Always 1 for single-node training
- `num_processes`: Number of GPUs available on the node
- `mixed_precision`: Precision mode (bf16, fp16, or no)
- `distributed_type`: MULTI_GPU for multi-GPU training

### Training Parameters

- `--per_device_train_batch_size`: Batch size per GPU (reduce if OOM)
- `--gradient_accumulation_steps`: Steps to accumulate gradients
- `--bf16`/`--fp16`: Enable mixed precision training
- `--dataloader_num_workers`: Number of data loading workers

## Memory Optimization

For large models or limited GPU memory:

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable mixed precision (`--bf16` or `--fp16`)
4. Reduce `max_prompt_length` and `max_completion_length`

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or enable mixed precision
2. **Process hanging**: Check GPU availability and CUDA setup
3. **Model loading errors**: Ensure all GPUs have access to model files

### Debug Mode

Enable debug mode in accelerate config:
```yaml
debug: true
```

## Performance Tips

1. Use BF16 for better performance on modern GPUs
2. Adjust `dataloader_num_workers` based on CPU cores
3. Use gradient checkpointing for memory efficiency
4. Monitor GPU utilization with `nvidia-smi`

## Example Commands

```bash
# Basic training with default settings
./launch_multigpu_training.sh

# Training with custom model and BF16
./launch_multigpu_training.sh --model_path /path/to/model --bf16

# Training with custom batch size and learning rate
./launch_multigpu_training.sh --learning_rate 2e-6 --per_device_train_batch_size 1
```
