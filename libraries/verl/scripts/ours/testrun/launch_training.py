#!/usr/bin/env python3
"""
Training launcher script for VERL PPO training.

This script provides a Python interface to launch training jobs with configurable parameters.
All default values are specified here and can be overridden via command-line arguments.
"""

import argparse
import os
import subprocess
import sys
from typing import Optional


def setup_environment():
    """Set up required environment variables for training."""
    env_vars = {
        # CUDA environment
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        
        # Ray environment
        "RAY_DEDUP_LOGS": "0",
        "RAY_IGNORE_UNHANDLED_ERRORS": "1",
        "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
        "RAY_DISABLE_IMPORT_WARNING": "1",
        
        # Other settings
        "HYDRA_FULL_ERROR": "1",
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("Environment variables configured successfully")


def parse_args():
    """Parse command-line arguments with defaults."""
    parser = argparse.ArgumentParser(
        description="Launch VERL PPO training with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen3-8B",
                        help="Model name")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model (default: /fast/nchandak/models/{model_name})")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--kl_coeff", type=float, default=0.005,
                        help="KL coefficient")
    parser.add_argument("--clip_ratio_low", type=float, default=0.2,
                        help="Lower clip ratio")
    parser.add_argument("--clip_ratio_high", type=float, default=0.28,
                        help="Higher clip ratio")
    parser.add_argument("--clip_ratio_c", type=float, default=10.0,
                        help="Clip ratio C")
    
    # Sequence lengths
    parser.add_argument("--max_prompt_length", type=int, default=4096,
                        help="Maximum prompt length")
    parser.add_argument("--max_response_length", type=int, default=4096,
                        help="Maximum response length")
    
    # Data configuration
    parser.add_argument("--train_files", type=str, 
                        default="/fast/nchandak/forecasting/datasets/verl/freeform/data70k-retrieval/ranked_queries_train_30_with_metaculus_randomk_shuffled.jsonl",
                        help="Training data files")
    parser.add_argument("--val_files", type=str,
                        default="/fast/nchandak/forecasting/datasets/verl/freeform/data70k-retrieval/validation_with_binary_5.jsonl",
                        help="Validation data files")
    parser.add_argument("--train_batch_size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=512,
                        help="Validation batch size")
    parser.add_argument("--shuffle", type=str, default="False",
                        help="Whether to shuffle data (True/False)")
    parser.add_argument("--filter_overlong_prompts", type=str, default="True",
                        help="Filter overlong prompts (True/False)")
    parser.add_argument("--truncation", type=str, default="error",
                        help="Truncation strategy")
    
    # Reward model configuration
    parser.add_argument("--reward_model_path", type=str,
                        default="/fast/nchandak/models/Qwen3-4B-Instruct-2507",
                        help="Path to reward model")
    parser.add_argument("--reward_strategy", type=str, default="matcher",
                        help="Reward strategy")
    parser.add_argument("--reward_manager", type=str, default="naive",
                        help="Reward manager type")
    parser.add_argument("--add_correctness", type=str, default="True",
                        help="Add correctness to reward (True/False)")
    parser.add_argument("--reward_micro_batch_size", type=int, default=0,
                        help="Reward model micro batch size")
    
    # Actor/Rollout configuration
    parser.add_argument("--ppo_mini_batch_size", type=int, default=256,
                        help="PPO mini batch size")
    parser.add_argument("--use_dynamic_bsz", type=str, default="True",
                        help="Use dynamic batch size (True/False)")
    parser.add_argument("--warmup_style", type=str, default="cosine",
                        help="Learning rate warmup style")
    parser.add_argument("--lr_warmup_steps_ratio", type=float, default=0.01,
                        help="Learning rate warmup steps ratio")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum learning rate ratio")
    parser.add_argument("--use_remove_padding", type=str, default="True",
                        help="Use remove padding (True/False)")
    parser.add_argument("--use_kl_loss", type=str, default="True",
                        help="Use KL loss (True/False)")
    parser.add_argument("--kl_loss_type", type=str, default="low_var_kl",
                        help="KL loss type")
    parser.add_argument("--enable_gradient_checkpointing", type=str, default="True",
                        help="Enable gradient checkpointing (True/False)")
    parser.add_argument("--param_offload", type=str, default="True",
                        help="FSDP parameter offload (True/False)")
    parser.add_argument("--optimizer_offload", type=str, default="True",
                        help="FSDP optimizer offload (True/False)")
    parser.add_argument("--ref_param_offload", type=str, default="True",
                        help="Reference model parameter offload (True/False)")
    
    # Rollout configuration
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1,
                        help="Tensor model parallel size")
    parser.add_argument("--rollout_name", type=str, default="vllm",
                        help="Rollout engine name")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                        help="GPU memory utilization for rollout")
    parser.add_argument("--rollout_n", type=int, default=8,
                        help="Number of samples for rollout")
    parser.add_argument("--load_format", type=str, default="safetensors",
                        help="Model load format")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--enable_chunked_prefill", type=str, default="True",
                        help="Enable chunked prefill (True/False)")
    
    # Validation rollout configuration
    parser.add_argument("--val_do_sample", type=str, default="True",
                        help="Validation: do sampling (True/False)")
    parser.add_argument("--val_temperature", type=float, default=0.6,
                        help="Validation temperature")
    parser.add_argument("--val_top_p", type=float, default=0.95,
                        help="Validation top-p")
    parser.add_argument("--val_n", type=int, default=1,
                        help="Validation number of samples")
    
    # Trainer configuration
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs per node")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("--project_name", type=str, default="rl-retrievalfixed-data70k",
                        help="Project name for logging")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (default: auto-generated)")
    parser.add_argument("--save_freq", type=int, default=50,
                        help="Save frequency (epochs)")
    parser.add_argument("--test_freq", type=int, default=60,
                        help="Test frequency (epochs)")
    parser.add_argument("--total_epochs", type=int, default=7,
                        help="Total training epochs")
    parser.add_argument("--critic_warmup", type=int, default=0,
                        help="Critic warmup epochs")
    parser.add_argument("--logger", type=str, default="console,wandb",
                        help="Loggers to use (comma-separated)")
    parser.add_argument("--val_before_train", type=str, default="False",
                        help="Validate before training (True/False)")
    parser.add_argument("--default_local_dir", type=str, default=None,
                        help="Default local directory for checkpoints")
    
    # Algorithm configuration
    parser.add_argument("--adv_estimator", type=str, default="grpo",
                        help="Advantage estimator")
    parser.add_argument("--norm_adv_by_std_in_grpo", type=str, default="False",
                        help="Normalize advantage by std in GRPO (True/False)")
    
    # Additional configuration
    parser.add_argument("--final_reward_manager", type=str, default="prime",
                        help="Final reward manager type")
    parser.add_argument("--taskset_cpus", type=str, default="0-31",
                        help="CPU cores to use with taskset")
    
    # Dry run option
    parser.add_argument("--dry_run", action="store_true",
                        help="Print command without executing")
    
    return parser.parse_args()


def build_command(args):
    """Build the training command with all arguments."""
    # Set model path if not provided
    if args.model_path is None:
        args.model_path = f"/fast/nchandak/models/{args.model_name}"
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_name}-{args.max_response_length}-{args.lr}-kl{args.kl_coeff}-shuffled"
    
    # Set default local directory if not provided
    if args.default_local_dir is None:
        args.default_local_dir = f"/lustre/scratch/nchandak/forecasting/training/verl/checkpoints/{args.project_name}/{args.experiment_name}"
    
    # Calculate max token length
    max_token_len = args.max_prompt_length + args.max_response_length
    
    # Build the command
    cmd = [
        "taskset", "-c", args.taskset_cpus,
        "python3", "-m", "verl.trainer.main_ppo",
    ]
    
    # Algorithm configuration
    cmd.extend([
        f"algorithm.adv_estimator={args.adv_estimator}",
        f"algorithm.norm_adv_by_std_in_grpo={args.norm_adv_by_std_in_grpo}",
    ])
    
    # Reward model configuration
    cmd.extend([
        "reward_model.enable=True",
        f"reward_model.model.path={args.reward_model_path}",
        f"reward_model.strategy={args.reward_strategy}",
        f"reward_model.reward_manager={args.reward_manager}",
        f"+reward_model.add_correctness={args.add_correctness}",
        f"reward_model.micro_batch_size={args.reward_micro_batch_size}",
    ])
    
    # Data configuration
    cmd.extend([
        f"data.train_files={args.train_files}",
        f"data.val_files={args.val_files}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"data.shuffle={args.shuffle}",
        f"data.filter_overlong_prompts={args.filter_overlong_prompts}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"data.truncation={args.truncation}",
    ])
    
    # Actor/Rollout/Ref configuration
    cmd.extend([
        f"actor_rollout_ref.model.path={args.model_path}",
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        f"actor_rollout_ref.actor.optim.warmup_style={args.warmup_style}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={args.lr_warmup_steps_ratio}",
        f"actor_rollout_ref.actor.optim.min_lr_ratio={args.min_lr_ratio}",
        f"actor_rollout_ref.model.use_remove_padding={args.use_remove_padding}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={args.use_dynamic_bsz}",
        f"actor_rollout_ref.actor.clip_ratio_low={args.clip_ratio_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={args.clip_ratio_high}",
        f"actor_rollout_ref.actor.clip_ratio_c={args.clip_ratio_c}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_token_len}",
        f"actor_rollout_ref.actor.use_kl_loss={args.use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_coeff}",
        f"actor_rollout_ref.actor.kl_loss_type={args.kl_loss_type}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={args.enable_gradient_checkpointing}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={args.param_offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={args.optimizer_offload}",
    ])
    
    # Rollout configuration
    cmd.extend([
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.name={args.rollout_name}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.load_format={args.load_format}",
        f"actor_rollout_ref.rollout.temperature={args.temperature}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={args.enable_chunked_prefill}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_token_len}",
    ])
    
    # Reference model configuration
    cmd.extend([
        f"actor_rollout_ref.ref.fsdp_config.param_offload={args.ref_param_offload}",
    ])
    
    # Validation rollout configuration
    cmd.extend([
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={args.val_do_sample}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={args.val_temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={args.val_top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.n={args.val_n}",
    ])
    
    # Trainer configuration
    logger_list = args.logger.split(',')
    logger_str = "[" + ",".join([f"'{l.strip()}'" for l in logger_list]) + "]"
    
    cmd.extend([
        f"trainer.critic_warmup={args.critic_warmup}",
        f"trainer.logger={logger_str}",
        f"++trainer.val_before_train={args.val_before_train}",
        f"trainer.n_gpus_per_node={args.num_gpus}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.default_local_dir={args.default_local_dir}",
    ])
    
    # Additional configuration
    cmd.extend([
        f"+reward_manager={args.final_reward_manager}",
    ])
    
    return cmd


def main():
    """Main entry point."""
    # Setup environment
    setup_environment()
    
    # Parse arguments
    args = parse_args()
    
    # Build command
    cmd = build_command(args)
    
    # Print configuration summary
    print("\n" + "="*80)
    print("Training Configuration Summary")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Project: {args.project_name}")
    print(f"Experiment: {args.experiment_name if args.experiment_name else 'auto-generated'}")
    print(f"Learning Rate: {args.lr}")
    print(f"KL Coefficient: {args.kl_coeff}")
    print(f"Max Prompt Length: {args.max_prompt_length}")
    print(f"Max Response Length: {args.max_response_length}")
    print(f"Training Batch Size: {args.train_batch_size}")
    print(f"Total Epochs: {args.total_epochs}")
    print(f"GPUs: {args.num_gpus}")
    print("="*80 + "\n")
    
    # Print command
    print("Executing command:")
    print(" \\\n  ".join(cmd))
    print()
    
    if args.dry_run:
        print("Dry run mode - command not executed")
        return 0
    
    # Execute command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

