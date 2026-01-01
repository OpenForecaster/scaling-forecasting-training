"""
GPT-OSS 20B GRPO Training Script
Based on: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb

Trains GPT-OSS 20B model using GRPO (Group Relative Policy Optimization) with Unsloth optimizations.

Usage:
# Run with default values
python gptoss_grpo.py

# Or override specific parameters
python gptoss_grpo.py \\
    --model_name_or_path /fast/nchandak/models/gpt-oss-20b \\
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \\
    --output_dir /lustre/scratch/nchandak/forecasting/training/gptoss_grpo \\
    --run_name gptoss-grpo-20b \\
    --learning_rate 1e-6 \\
    --lora_r 16 \\
    --lora_target_modules all-linear
"""

import argparse
import os
from datetime import datetime
from typing import Optional

import torch
from unsloth import FastLanguageModel  # Must be imported before transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
import wandb

from trl import GRPOConfig, GRPOTrainer
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-OSS 20B GRPO Training with Unsloth")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="/fast/nchandak/models/gpt-oss-20b-bf16",
                        help="Path to the model or model name")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="Path to the tokenizer (defaults to model_name_or_path)")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/OpenR1-Math-220k-default-verified",
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_train_split", type=str, default="train",
                        help="Dataset split to use for training")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to use for training")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/lustre/scratch/nchandak/forecasting/training/gptoss_grpo",
                        help="Output directory for the trained model")
    parser.add_argument("--run_name", type=str, default="gptoss-grpo-20b",
                        help="Name for the wandb run")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=2048,
                        help="Maximum completion length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Logging steps")
    parser.add_argument("--num_generations", type=int, default=2,
                        help="Number of generations")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="LoRA target modules")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    
    # Unsloth arguments
    parser.add_argument("--use_unsloth", action="store_true", default=True,
                        help="Use Unsloth optimizations")
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                        help="Load model in 4-bit quantization")
    
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


def simple_reward_function(completions, **kwargs):
    """Simple reward function for GRPO training."""
    rewards = []
    for completion in completions:
        # Simple length-based reward (can be replaced with more sophisticated logic)
        content = completion[0]["content"] if isinstance(completion, list) else completion
        # Reward longer, more detailed responses
        reward = min(len(content) / 1000.0, 1.0)  # Normalize to 0-1
        rewards.append(reward)
    return rewards


if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables for better stability
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_TIMEOUT"] = "1800"
    
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
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    ################
    # Setup wandb (only on main process)
    ################
    if accelerator.is_main_process:
        # Generate timestamped run name if using default
        if args.run_name == "gptoss-grpo-20b":
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"gptoss-grpo-{args.model_name_or_path.split('/')[-1]}-{now}"
        
        wandb.init(
            project="gptoss-grpo",
            name=args.run_name,
            config=vars(args)
        )

    ################
    # Load model with Unsloth
    ################
    if args.use_unsloth:
        # Load model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=args.max_prompt_length + args.max_completion_length,
            dtype=None,  # Auto-detect
            load_in_4bit=args.load_in_4bit,
            device_map="balanced",
            # device_map removed for distributed training - Accelerate handles this
        )
        
        # Configure LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=args.lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            # bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            # use_rslora=False,
            # loftq_config=None,
        )
    else:
        print("Loading model without Unsloth")
        # Standard loading without Unsloth
        tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Setup LoRA config
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
        )

    ################
    # Load dataset
    ################
    dataset = load_dataset(args.dataset_name, split=args.dataset_train_split)
    
    # Limit samples for faster training
    if len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    def format_conversation(example):
        """Format the dataset for GRPO training with boxed answer instruction."""
        # OpenR1-Math format - add instruction to output in boxed format
        problem_text = example["problem"] + "\n\nOutput your final answer in \\boxed{} format."
        prompt = [{"role": "user", "content": problem_text}]
        return {
            "prompt": prompt,
            "solution": example["solution"],
            "reasoning_effort": "low"
        }

    dataset = dataset.map(format_conversation)

    # Remove unnecessary columns
    columns_to_keep = ["prompt", "solution", "reasoning_effort"]
    
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    ################
    # Setup training config
    ################
    # training_args = GRPOConfig(
    #     output_dir=args.output_dir,
    #     learning_rate=args.learning_rate,
    #     num_train_epochs=args.num_train_epochs,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     max_prompt_length=args.max_prompt_length,
    #     max_completion_length=args.max_completion_length,
    #     warmup_ratio=args.warmup_ratio,
    #     logging_steps=args.logging_steps,
    #     report_to=["wandb"],
    #     push_to_hub=args.push_to_hub,
    #     save_steps=100,
    #     eval_steps=100,
    #     evaluation_strategy="steps",
    # )
    
    training_args = GRPOConfig(
        temperature = 1.0,
        learning_rate = args.learning_rate,
        weight_decay = 0.01,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = args.logging_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_generations = args.num_generations,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        max_steps = 100,
        save_steps = 100,
        report_to = "wandb" if accelerator.is_main_process else [],
        output_dir = args.output_dir,
        dataloader_num_workers = args.dataloader_num_workers,
        bf16 = args.bf16,
        fp16 = args.fp16,
        # Distributed training settings for stability
        ddp_find_unused_parameters = False,
        remove_unused_columns = False,
        # NCCL settings for better stability
        ddp_backend = "nccl",
        ddp_timeout = 1800,  # 30 minutes timeout
        local_rank = accelerator.local_process_index,
    )


    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[strip_reasoning_accuracy_reward],
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    # Start training with error handling
    try:
        print(f"Starting GRPO training with {len(dataset)} samples...")
        trainer.train()

        # Wait for all processes to finish training
        accelerator.wait_for_everyone()

        # Save model (only on main process)
        if accelerator.is_main_process:
            print(f"Saving model to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            if args.push_to_hub:
                trainer.push_to_hub(dataset_name=args.dataset_name)
        
        # Finalize wandb (only on main process)
        if accelerator.is_main_process:
            wandb.finish()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        if accelerator.is_main_process:
            wandb.finish()
        raise e
