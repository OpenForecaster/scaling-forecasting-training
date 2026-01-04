#!/usr/bin/env python
# coding: utf-8

"""
Evaluation script for GPQA (Graduate-Level Google-Proof Q&A) benchmark.
Evaluates models on challenging graduate-level multiple-choice questions.
Tests expert-level reasoning across science domains.
Uses vLLM for efficient inference.
"""

import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Optional, List, Tuple
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm 
import json
import os 
import logging
import time 
import sys
from dataclasses import dataclass
import random

# Import vLLM for faster generation
from vllm import LLM, SamplingParams

# Set SEED
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
# Set cuDNN for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables to control threading for various libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_DIR = ""
DATA_SPLIT = "train"
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/gpqa/"
DATA = "gpqa"

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str = "main"
    torch_dtype: Optional[str] = "bfloat16"
    trust_remote_code: bool = True

@dataclass
class EvalScriptArguments:
    dataset_id_or_path: str = "Idavidrein/gpqa"
    dataset_config: str = "gpqa_diamond"
    dataset_splits: str = "train"
    tokenizer_name_or_path: Optional[str] = None
    model_checkpoint: str = None
    per_device_eval_batch_size: int = 32
    output_dir: str = "results/"

def add_idx_column(dataset: Dataset) -> Dataset:
    """
    Adds an 'idx' column to the dataset, storing the original row index.
    """
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)

def extract_answer(completion: str) -> Optional[str]:
    """
    Extracts the final answer from the LLM's output.
    Returns the extracted answer letter (A, B, C, D) or None if not found.
    """
    if not completion:
        return None
        
    # Try to extract from <answer> tags - use findall to get all matches and take the last one
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if answer_matches:
        # Extract just the letter from the last match
        answer = answer_matches[-1].strip()
        # If the answer contains a single letter, return it
        if re.match(r'^[A-D]$', answer):
            return answer
        # Try to extract just the letter if it's in a format like "A." or "Option A"
        letter_match = re.search(r'([A-D])[\.:\)]|[Oo]ption\s+([A-D])', answer)
        if letter_match:
            return letter_match.group(1) if letter_match.group(1) else letter_match.group(2)
    
    # Look for patterns like "The answer is A" or "I choose B"
    answer_phrases = [
        r'the answer is\s+([A-D])',
        r'I choose\s+([A-D])',
        r'answer:\s+([A-D])',
        r'final answer:?\s+([A-D])',
        r'option\s+([A-D])'
    ]
    
    for phrase in answer_phrases:
        matches = re.findall(phrase, completion, re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # Return the last match, ensuring it's uppercase
    
    # Last resort: look for any standalone capital letter that might be an answer
    standalone_letters = re.findall(r'(?:^|\s)([A-D])(?:$|\s|\.|\))', completion)
    if standalone_letters:
        return standalone_letters[-1]
    
    return None

def extract_probability(completion: str) -> Optional[float]:
    """
    Extracts the probability from the LLM's output.
    Returns the probability as a float.
    """
    matches = re.finditer(r"<probability>(.*?)<\/probability>", completion, re.DOTALL)
    matches_list = list(matches)

    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    probability_text = last_match.group(1).strip()

    # Try to parse probability as float
    try:
        probability = float(probability_text)
        return probability
    except (ValueError, TypeError):
        return None 

def format_mcq_prompt(
    question: str,
    options: List[str],
    category: str = "",
) -> str:
    """
    Format the multiple choice question prompt.
    """
    # Format options as A, B, C, D
    options_str = ""
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, ...
        options_str += f"{letter}. {option}\n"
    
    category_text = f" in {category}" if category else ""
    
    prompt = f"""You will be asked a multiple-choice question{category_text}. Please provide your reasoning before stating your final answer. Also express how confident you are in your answer. You will be scored based on correctness of your answer and also using the multi-class brier scoring rule (mean squared error) for the probability of you assign to your answer.

Question: {question}

Options:
{options_str}

Think step by step and put your final answer in <answer> </answer> tags and your probability in that in <probability> </probability> tags.
Your final answer should be a SINGLE LETTER in uppercase (A, B, C, D, etc.) corresponding to the correct option.
Your response SHOULD STRICTLY END with your answer choice in <answer> </answer> tags and your probability in <probability> </probability> tags.
"""

    return prompt

def load_model_and_tokenizer(model_path: str, model_name: str = None):
    """
    Load vLLM model and tokenizer
    """
    if model_name is None:
        model_name = model_path.rstrip("/").split("/")[-1]
    logger.info(f"Using model_name: {model_name}")

    logger.info(f"Loading model with vLLM from local directory: {model_path}")
    
    # Initialize vLLM model
    try:
        # Load tokenizer separately for prompt processing
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Check if model is multimodal (like Llama-4-Scout)
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            is_multimodal = hasattr(config, 'vision_config') or 'vision' in str(config).lower()
            logger.info(f"Detected multimodal model: {is_multimodal}")
        except:
            is_multimodal = False
        
        # Use bfloat16 for better compatibility, especially with multimodal models
        dtype = "auto" #  "bfloat16"
        
        # Initialize vLLM model with tensor parallelism
        vllm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "dtype": dtype,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": torch.cuda.device_count(),
        }
        
        # For multimodal models, we might need different settings
        if is_multimodal:
            logger.warning("Detected multimodal model. This may not be fully supported by vLLM.")
            # Reduce GPU memory utilization for multimodal models
            vllm_kwargs["gpu_memory_utilization"] = 0.75
            # Try to disable vision processing if possible
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 0}
        
        model = LLM(**vllm_kwargs)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying alternative loading approach...")
        
        # Alternative approach: try different dtypes and settings
        for dtype in ["bfloat16", "float16", "auto"]:
            try:
                logger.info(f"Attempting to load with dtype: {dtype}")
                model = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    dtype=dtype,
                    gpu_memory_utilization=0.75,
                    tensor_parallel_size=1,  # Use single GPU to avoid multi-GPU issues
                    enforce_eager=True,  # Use eager mode for better compatibility
                )
                logger.info(f"Successfully loaded model with dtype: {dtype}")
                break
            except Exception as inner_e:
                logger.warning(f"Failed with dtype {dtype}: {inner_e}")
                if dtype == "auto":  # Last attempt
                    raise RuntimeError(f"Could not load model with any dtype. Last error: {inner_e}")
        
    return model, tokenizer

def preprocess_dataset(dataset, dataset_config: str = "gpqa_diamond"):
    """
    Preprocess the GPQA dataset to format it consistently
    """
    processed_data = []
    
    for i, row in enumerate(dataset):
        # Extract the main fields
        question = row.get("Question", "")
        correct_answer = row.get("Correct Answer", "")
        incorrect_1 = row.get("Incorrect Answer 1", "")
        incorrect_2 = row.get("Incorrect Answer 2", "")
        incorrect_3 = row.get("Incorrect Answer 3", "")
        record_id = row.get("Record ID", f"gpqa_{i}")
        subdomain = row.get("Subdomain", "")
        
        # Create options list and shuffle to randomize correct answer position
        options = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
        
        # Remove any None or empty options
        options = [opt.strip() for opt in options if opt and opt.strip()]
        
        if len(options) < 2:  # Skip if we don't have enough options
            continue
            
        # Shuffle options to randomize correct answer position
        shuffled_indices = list(range(len(options)))
        random.shuffle(shuffled_indices)
        
        shuffled_options = [options[i] for i in shuffled_indices]
        # Find new position of correct answer
        correct_answer_index = shuffled_indices.index(0)  # 0 was the original position of correct answer
        
        processed_row = {
            "idx": i,
            "question": question,
            "options": shuffled_options,
            "answer_index": correct_answer_index,
            "answer": chr(65 + correct_answer_index),  # Convert to A, B, C, D
            "record_id": record_id,
            "category": subdomain,
            "dataset": dataset_config
        }
        
        processed_data.append(processed_row)
    
    return Dataset.from_list(processed_data)

def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset,
    prompt_fn=format_mcq_prompt,
    max_new_tokens: int = 4096,
    batch_size: int = 32,
    max_prompt_length: int = 4096,
    num_generations: int = 5,  # Multiple generations per prompt
    temperature: float = 0.6,
    top_p: float = 0.95,
):
    """
    Run batched inference with multiple generations per prompt using vLLM
    """
    # Create prompts from each row
    all_prompts = []
    all_idxs = []
    all_row_data = []
    
    for row in dataset:
        # Format the prompt for each example    
        local_prompt = prompt_fn(
            question=row["question"],
            options=row["options"],
            category=row.get("category", ""),
        )
        try:
            chat = [
            {
                "role": "user",
                "content": local_prompt,
            },
            ]
            if 'qwen3' in model_name.lower():
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, 
                                                        add_generation_prompt=True, enable_thinking=True)
            else:
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        except Exception as e:
            logger.info(f"Error in tokenizer.apply_chat_template: {e}")
            prompt = local_prompt
            
        all_prompts.append(prompt)
        all_idxs.append(row["idx"])
        all_row_data.append(row)
    
    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=num_generations,  # Number of generations per prompt
    )
    
    # Process all prompts with vLLM
    logger.info(f"Starting generation with vLLM for {len(all_prompts)} prompts, {num_generations} generations each")
    start_time = time.time()
    
    # Generate completions using vLLM's batched API
    all_outputs = model.generate(all_prompts, sampling_params)
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Process results - group by prompt instead of individual generations
    all_results = []
    correct_count = 0
    total_count = 0
    
    for i, outputs in enumerate(all_outputs):
        prompt = all_prompts[i]
        idx = all_idxs[i]
        row = all_row_data[i]
        
        # Collect all generations for this prompt
        responses = []
        completion_tokens_list = []
        extracted_answers = []
        
        for output in outputs.outputs:
            generated_text = output.text
            
            # Find where the completion begins
            if "</think>" in generated_text:
                answer = generated_text.split("</think>")[1]
            else:
                answer = generated_text
                
            # Calculate token counts (approximate for vLLM)
            completion_tokens = len(tokenizer.encode(answer))
            
            # Extract answer
            extracted_answer = extract_answer(answer)
            final_prob = extract_probability(answer)
            final_ans = {extracted_answer: final_prob}
            
            responses.append(answer)
            completion_tokens_list.append(completion_tokens)
            extracted_answers.append(final_ans)

        # Calculate prompt tokens once per prompt
        prompt_tokens = len(tokenizer.encode(prompt))
        
        # Determine if any of the generations got the correct answer
        correct_answer = row["answer"]
        any_correct = any(list(ans.keys())[0] == correct_answer for ans in extracted_answers if ans is not None)
        
        # For majority voting - count most frequent valid answer
        valid_answers = [list(ans.keys())[0] for ans in extracted_answers if ans is not None]
        if valid_answers:
            from collections import Counter
            answer_counts = Counter(valid_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            majority_correct = majority_answer == correct_answer
        else:
            majority_answer = None
            majority_correct = False
        
        if any_correct:
            correct_count += 1
        total_count += 1
        
        # Store result with lists for generations
        result = {
            "model": model_name,
            "split": DATA_SPLIT,
            "data_type": DATA,
            "idx": idx,
            "response": responses,  # List of responses
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens_list,  # List of completion token counts
            "extracted_answer": extracted_answers,  # List of extracted answers
            "majority_answer": majority_answer,
            "any_correct": any_correct,
            "majority_correct": majority_correct,
            # Additional fields from the dataset
            "question": row.get("question", ""),
            "options": row.get("options", []),
            "answer": row.get("answer", ""),
            "answer_index": row.get("answer_index", -1),
            "record_id": row.get("record_id", ""),
            "category": row.get("category", ""),
        }
        
        all_results.append(result)
    
    # Log accuracy statistics
    accuracy = correct_count / total_count if total_count > 0 else 0
    majority_accuracy = sum(r["majority_correct"] for r in all_results) / total_count if total_count > 0 else 0
    
    logger.info(f"Any-correct accuracy: {correct_count}/{total_count} ({accuracy*100:.2f}%)")
    logger.info(f"Majority-vote accuracy: {sum(r['majority_correct'] for r in all_results)}/{total_count} ({majority_accuracy*100:.2f}%)")
    
    # Log mean output token length with standard deviation
    all_completion_tokens = []
    for result in all_results:
        all_completion_tokens.extend(result["completion_tokens"])
    mean_output_length = np.mean(all_completion_tokens)
    std_output_length = np.std(all_completion_tokens)
    logger.info(f"Mean output token length: {mean_output_length:.2f} ± {std_output_length:.2f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/gpqa/", help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-4B", help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=32768, help="Maximum number of new tokens for generation")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for generation")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for generation")
    
    parser.add_argument('--data_split', type=str, default="train", help="Data split to use")
    
    parser.add_argument('--dataset', type=str, default="Idavidrein/gpqa",
                      help="HuggingFace dataset path")
    parser.add_argument('--dataset_config', type=str, default="gpqa_diamond",
                      help="Dataset configuration/subset to use")
    
    parser.add_argument('--num_generations', type=int, default=5, help="Number of generations to use per prompt")
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_config  # Use config name for file naming
    
    # Create output directory structure
    output_base_dir = os.path.join(args.base_save_dir, dataset_name)
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")

    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    MODEL_DIR = args.model_dir
    DATA_SPLIT = args.data_split
    DATA = dataset_name
    
    # Load dataset from HuggingFace
    logger.info(f"Loading dataset {args.dataset} with config {args.dataset_config}")
    dataset = load_dataset(args.dataset, args.dataset_config)[args.data_split]
    
    # Preprocess the dataset to standardize format
    dataset = preprocess_dataset(dataset, args.dataset_config)
    dataset = add_idx_column(dataset)
    logger.info(f"Dataset size: {len(dataset)}")
    
    new_tokens = args.max_new_tokens
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {new_tokens}")
    logger.info(f"Model directory: {MODEL_DIR}")
    
    model_name = args.model
    
    # Extract model name from model_dir 
    if args.model == "None":
        model_name = MODEL_DIR.rstrip("/").split("/")[-1]
        if "__" in model_name:
            model_name = model_name.split("__")[1]
        
    logger.info(f"Model name: {model_name}")
    
    output_file = os.path.join(
        output_base_dir,
        f"{model_name}_{DATA_SPLIT}_size_{len(dataset)}_generations_{args.num_generations}.jsonl"
    )
    logger.info(f"Output file: {output_file}")
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Exiting without running evaluation.")
        exit(0)

    
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Run evaluation
    all_results = evaluate_model(
        model_name, 
        model, 
        tokenizer, 
        dataset, 
        prompt_fn=format_mcq_prompt,
        max_new_tokens=new_tokens, 
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Save results as JSONL
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')

    logger.info(f"Saved {len(all_results)} question results to {output_file}")
    
    # Log final statistics
    total_generations = len(all_results) * args.num_generations
    valid_extractions = sum(
        sum(1 for ans in result['extracted_answer'] if ans is not None) 
        for result in all_results
    )
    
    any_correct_count = sum(result['any_correct'] for result in all_results)
    majority_correct_count = sum(result['majority_correct'] for result in all_results)
    
    logger.info(f"Valid answer extractions: {valid_extractions}/{total_generations} ({valid_extractions/total_generations*100:.1f}%)")
    logger.info(f"Final any-correct accuracy: {any_correct_count}/{len(all_results)} ({any_correct_count/len(all_results)*100:.1f}%)")
    logger.info(f"Final majority-vote accuracy: {majority_correct_count}/{len(all_results)} ({majority_correct_count/len(all_results)*100:.1f}%)") 