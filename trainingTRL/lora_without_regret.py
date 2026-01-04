"""
LoRA Without Regret - GRPO Training Script
Based on: https://thinkingmachines.ai/blog/lora/

Trains a language model using GRPO with LoRA adapters for parameter-efficient 
fine-tuning on mathematical reasoning tasks.

Usage:
# Run with default values
python lora_without_regret.py

# Or override specific parameters
python lora_without_regret.py \\
    --model_name_or_path /fast/nchandak/models/Qwen/Qwen3-0.6B \\
    --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified \\
    --output_dir grpo-lora-qwen3-0.6b \\
    --run_name grpo-lora-qwen3-0.6b \\
    --learning_rate 1e-6 \\
    --lora_r 16 \\
    --lora_target_modules all-linear
"""

import argparse
import os
from typing import Optional
from datetime import datetime

import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed

from trl import GRPOConfig, GRPOTrainer

################
# Reward Function for Training
################

def strip_reasoning_accuracy_reward(
    completions: list[list[dict[str, str]]], solution: list[str], **kwargs
) -> list[Optional[float]]:
    """Reward function that strips reasoning tags and checks mathematical accuracy.

    This function:
    1. Extracts the content from completions
    2. Removes <think></think> tags (for reasoning that shouldn't be evaluated)
    3. Parses both the gold solution and the predicted answer
    4. Uses math_verify to check if they are mathematically equivalent

    Args:
        completions: List of model completions, each containing a list of messages
        solution: List of ground truth solutions
        **kwargs: Additional arguments (ignored but required for trainer compatibility)

    Returns:
        List of rewards where:
        - 1.0 if the answer is correct
        - 0.0 if the answer is incorrect
        - None if the solution is not parseable (skips this example)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Strip reasoning tags from completion
        while "<think>" in content and "</think>" in content:
            start = content.find("<think>")
            end = content.find("</think>", start)
            if start != -1 and end != -1:
                content = content[:start] + content[end + len("</think>") :]
            else:
                break

        # Parse gold solution
        gold_parsed = parse(
            f"${sol}$",
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, try_extract_without_anchor=True
                )
            ],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0,
                        normalization_config=NormalizationConfig(
                            basic_latex=True,
                            units=True,
                            malformed_operators=False,
                            nits=False,
                            boxed=True,
                        ),
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(
                    f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
                )
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None

        rewards.append(reward)

    return rewards

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Without Regret - GRPO Training")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="/fast/nchandak/models/Qwen3-1.7B",
                        help="Path to the model or model name")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Path to the tokenizer (defaults to model_name_or_path)")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/OpenR1-Math-220k-default-verified",
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_train_split", type=str, default="train",
                        help="Dataset split to use for training")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/lustre/scratch/nchandak/forecasting/training/lora_without_regret/qwen3-1.7b",
                        help="Output directory for the trained model")
    parser.add_argument("--run_name", type=str, default="grpo-lora-qwen3-1.7b",
                        help="Name for the wandb run")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Logging steps")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="LoRA target modules")
    
    # Distributed training arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 precision")
    
    # Other arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to hub after training")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    ################
    # Initialize Accelerate
    ################
    accelerator = Accelerator()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Print accelerator state for debugging
    print(f"Accelerator state: {accelerator.state}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")
    print(f"Device: {accelerator.device}")
    
    ################
    # Setup wandb (only on main process)
    ################
    if accelerator.is_main_process:
        # Generate timestamped run name if using default
        if args.run_name == "grpo-lora-qwen3-0.6b":
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"grpo-lora-{args.model_name_or_path.split('/')[-1]}-{now}"
        
        wandb.init(
            project="lora_without_regret",
            name=args.run_name,
            config=vars(args)
        )

    ################
    # Load tokenizer
    ################
    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Load dataset
    ################
    dataset = load_dataset(args.dataset_name, split=args.dataset_train_split)

    # Limit to 5k samples for faster training
    if len(dataset) > 5000:
        dataset = dataset.select(range(5000))

    def make_conversation(example):
        prompt = [{"role": "user", "content": example["problem"]}]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Remove unnecessary columns
    columns_to_remove = [
        col for col in dataset.column_names if col not in ["prompt", "solution"]
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    ################
    # Setup LoRA config
    ################
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
    )

    ################
    # Setup training config
    ################
    # Determine mixed precision
    mixed_precision = "no"
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["wandb"] if accelerator.is_main_process else [],
        # Distributed training settings
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=training_args,
        reward_funcs=[strip_reasoning_accuracy_reward],
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Start training
    trainer.train()

    # Wait for all processes to finish training
    accelerator.wait_for_everyone()

    # Save model (only on main process)
    if accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        if args.push_to_hub:
            trainer.push_to_hub(dataset_name=args.dataset_name)
    
    # Finalize wandb (only on main process)
    if accelerator.is_main_process:
        wandb.finish()