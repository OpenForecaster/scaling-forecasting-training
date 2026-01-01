#!/usr/bin/env python
# coding: utf-8

"""
Multiple-choice evaluation script for MMLU-Pro benchmark.
Alternative implementation of MMLU-Pro evaluation with MCQ-specific formatting.
Extracts answer choices and calculates accuracy on TIGER-Lab/MMLU-Pro dataset.
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
DATA_SPLIT = "test"
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/mmlu_pro/"
DATA = "mmlu_pro"

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: str = "main"
    torch_dtype: Optional[str] = "bfloat16"
    trust_remote_code: bool = True

@dataclass
class EvalScriptArguments:
    dataset_id_or_path: str = "TIGER-Lab/MMLU-Pro"
    dataset_splits: str = "test"
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
    Returns the extracted answer letter (A-J) or None if not found.
    """
    if not completion:
        return None
        
    # Try to extract from <answer> tags - use findall to get all matches and take the last one
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if answer_matches:
        # Extract just the letter from the last match
        answer = answer_matches[-1].strip()
        # If the answer contains a single letter, return it
        if re.match(r'^[A-J]$', answer):
            return answer
        # Try to extract just the letter if it's in a format like "A." or "Option A"
        letter_match = re.search(r'([A-J])[\.:\)]|[Oo]ption\s+([A-J])', answer)
        if letter_match:
            return letter_match.group(1) if letter_match.group(1) else letter_match.group(2)
    
    # Look for patterns like "The answer is A" or "I choose B"
    answer_phrases = [
        r'the answer is\s+([A-J])',
        r'I choose\s+([A-J])',
        r'answer:\s+([A-J])',
        r'final answer:?\s+([A-J])',
        r'option\s+([A-J])'
    ]
    
    for phrase in answer_phrases:
        matches = re.findall(phrase, completion, re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # Return the last match, ensuring it's uppercase
    
    # Last resort: look for any standalone capital letter that might be an answer
    standalone_letters = re.findall(r'(?:^|\s)([A-J])(?:$|\s|\.|\))', completion)
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
    # Format options as A, B, C, D, E, F, G, H, I, J
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
Your final answer should be a SINGLE LETTER in uppercase (A, B, C, D, E, F, G, H, I, J) corresponding to the correct option.
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

def preprocess_dataset(dataset, subset_name: str = "mmlu_pro"):
    """
    Preprocess the MMLU-Pro dataset to format it consistently
    """
    processed_data = []
    
    for i, row in enumerate(dataset):
        # Extract the main fields
        question = row.get("question", "")
        options = row.get("options", [])
        answer_index = row.get("answer_index", 0)
        answer = row.get("answer", "A")
        question_id = row.get("question_id", i)
        category = row.get("category", "")
        
        # Ensure we have at least 2 options
        if len(options) < 2:
            continue
            
        # MMLU-Pro has 10 options, so no need to shuffle - keep original order
        # Convert answer_index to letter if needed
        if isinstance(answer, int):
            answer = chr(65 + answer)  # Convert 0->A, 1->B, etc.
        elif not isinstance(answer, str):
            answer = chr(65 + answer_index)
        
        processed_row = {
            "idx": i,
            "question": question,
            "options": options,
            "answer_index": answer_index,
            "answer": answer,
            "question_id": question_id,
            "category": category,
            "dataset": subset_name
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
            "question_id": row.get("question_id", ""),
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
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/mmlu_pro/", help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-4B", help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=8192, help="Maximum number of new tokens for generation")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for generation")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for generation")
    
    parser.add_argument('--data_split', type=str, default="test", help="Data split to use")
    
    parser.add_argument('--dataset', type=str, default="TIGER-Lab/MMLU-Pro",
                      help="HuggingFace dataset path")
    
    parser.add_argument('--num_generations', type=int, default=3, help="Number of generations to use per prompt")
    
    # Add subset selection for testing
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    dataset_name = "mmlu_pro"  # Use simplified name for file naming
    
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
    logger.info(f"Loading dataset {args.dataset}")
    try:
        # Load using streaming first to avoid the parquet issue
        dataset_stream = load_dataset(args.dataset, streaming=True)
        dataset_iter = iter(dataset_stream[args.data_split])
        
        # Convert to list with size limit for memory efficiency
        max_samples = args.max_samples or 500  # Default to 500 for testing
        max_samples = 20000
        dataset_list = []
        for i, sample in enumerate(dataset_iter):
            if i >= max_samples:
                break
            dataset_list.append(sample)
        
        # Convert to Dataset object
        dataset = Dataset.from_list(dataset_list)
        logger.info(f"Successfully loaded {len(dataset)} samples using streaming")
        
    except Exception as e:
        logger.warning(f"Streaming failed: {e}")
        logger.info("Attempting to load a subset directly...")
        
        # Fallback: try to load smaller subset
        dataset = load_dataset(args.dataset, split=f"{args.data_split}[:500]")
        logger.info(f"Loaded {len(dataset)} samples directly")
    
    # Preprocess the dataset to standardize format
    dataset = preprocess_dataset(dataset, "mmlu_pro")
    dataset = add_idx_column(dataset)
    logger.info(f"Dataset size after preprocessing: {len(dataset)}")
    
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