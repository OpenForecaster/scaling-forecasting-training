"""
Evaluation script for FutureBench benchmark.
Evaluates models on future-oriented forecasting questions from the FutureBench dataset.
Tests ability to make predictions about future events across various domains.
Uses vLLM for efficient inference.
"""

import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, load_dataset
from typing import Optional, List, Tuple, Callable
from accelerate import Accelerator
from transformers import AutoTokenizer
from tqdm import tqdm 
import json
import os 
import logging
import time 
import sys
import random

# Import vLLM for faster generation
from vllm import LLM, SamplingParams

# Set SEED
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
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
ZERO_SHOT = True
OUTPUT_DIR = "/fast/nchandak/forecasting/evals/futurebench/"
DATA = "futurebench"

def add_idx_column(dataset: Dataset) -> Dataset:
    """
    Adds an 'idx' column to the dataset, storing the original row index.
    """
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)

def extract_answer(completion: str) -> Optional[str]:
    """
    Extracts the final answer from the LLM's output.
    Returns the raw answer text without type conversion.
    """
    matches = re.finditer(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
    matches_list = list(matches)
    
    if not matches_list:
        return None
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text

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

def format_futurebench_prompt(
    question: str,
    event_type: str = "",
    open_to_bet_until: str = ""
) -> str:
    """
    Format the prompt for FutureBench dataset.
    """
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best guess for the final answer (YES/NO). Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question: {question}

{f"Event Type: {event_type}" if event_type else ""}
{f"Open to bet until: {open_to_bet_until}" if open_to_bet_until else ""}

Think step by step about the information provided, reason about uncertainty and put your final answer (YES or NO) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. Thus, the range of the score is [-1, 0]. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be either YES or NO and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

    return prompt



def format_futurebench_prompt_binary(
    question: str,
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a BINARY forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your probability estimate of it resolving YES).
        
Question: {question}

Think step by step about the information provided, reason about uncertainty and put your final probability for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt

def load_model_and_tokenizer(model_path: str, model_name: str = None):
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
        num_gpus = torch.cuda.device_count()
        
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

def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset,
    max_new_tokens: int = 8192,
    batch_size: int = 32,
    max_prompt_length: int = 4096,
    num_generations: int = 1,  # Default to 1 for futurebench
):
    """
    Run batched inference with multiple generations per prompt using vLLM
    """
    # Create prompts from each row
    all_prompts = []
    all_idxs = []
    all_row_data = []
    
    for i, row in enumerate(dataset):
        # Format the prompt for each example
        local_prompt = format_futurebench_prompt_binary(
            question=row["question"],
        )

        # Defensive: if tokenizer is None, just use the raw prompt
        if tokenizer is None:
            logger.info("Warning: tokenizer is None, using raw prompt.")
            prompt = local_prompt
        else:
            try:
                chat = [
                    {
                        "role": "user",
                        "content": local_prompt,
                    }
                ]
                if 'qwen3' in (model_name or "").lower():
                    prompt = tokenizer.apply_chat_template(
                        chat, tokenize=False, 
                        add_generation_prompt=True, enable_thinking=True
                    )
                else:
                    prompt = tokenizer.apply_chat_template(
                        chat, tokenize=False, continue_final_message=True
                    )
            except Exception as e:
                logger.info(f"Error in tokenizer.apply_chat_template: {e}")
                prompt = local_prompt

        all_prompts.append(prompt)
        
        if i < 1:
            logger.info(f"Prompt: {prompt}")
            
        all_idxs.append(row["idx"])
        all_row_data.append(row)
    
    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
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
    
    for i, outputs in enumerate(all_outputs):
        prompt = all_prompts[i]
        idx = all_idxs[i]
        row = all_row_data[i]
        
        # Collect all generations for this prompt
        responses = []
        completion_tokens_list = []
        final_answers = []
        
        for output in outputs.outputs:
            generated_text = output.text
            
            # Find where the prompt ends and the completion begins
            # For futurebench, we'll use the full generated text as the answer
            answer = generated_text
            
            # Calculate token counts (approximate for vLLM)
            completion_tokens = len(tokenizer.encode(answer))
            
            final_prob = extract_probability(answer)
            last_ans = "YES"
            if "<answer>" in answer:
                last_ans = extract_answer(answer)
                
            final_ans = {last_ans: final_prob}
                
            responses.append(answer)
            completion_tokens_list.append(completion_tokens)
            final_answers.append(final_ans)

        # Calculate prompt tokens once per prompt
        prompt_tokens = len(tokenizer.encode(prompt))
        
        # Store result with lists for generations
        result = {
            "model": model_name,
            "split": DATA_SPLIT,
            "data_type": DATA,
            "idx": idx,
            "response": responses,  # List of responses
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens_list,  # List of completion token counts
            "extracted_answer": final_answers,  # List of final answers
            # Additional fields from futurebench
            "event_id": row.get("event_id", ""),
            "question": row.get("question", ""),
            "event_type": row.get("event_type", ""),
            "open_to_bet_until": row.get("open_to_bet_until", ""),
            "result": row.get("result", ""),
            "source": row.get("source", ""),
            # Convert result to binary label (1 for Yes, 0 for No/None of the above)
            "resolution": 1 if row.get("result", "").lower() == "yes" else 0,
        }
        
        all_results.append(result)
    
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
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/futurebench", help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-8B", help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name")
    
    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=16384, help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_split', type=str, default="train", help="Data split to use")
    
    parser.add_argument('--num_generations', type=int, default=5, help="Number of generations to use per prompt")
    
    args = parser.parse_args()
    
    # Extract dataset info
    dataset_name = f"futurebench"
    
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
    logger.info(f"Loading FutureBench dataset")
    dataset = load_dataset('futurebench/data', split=args.data_split)
    
    logger.info(f"Original dataset size: {len(dataset)}")
    
    # Keep only unique event_ids (take first occurrence of each event_id)
    seen_event_ids = set()
    unique_events = []
    
    for item in dataset:
        event_id = item['event_id']
        if event_id not in seen_event_ids:
            seen_event_ids.add(event_id)
            unique_events.append(item)
    
    logger.info(f"Unique events found: {len(unique_events)}")
    
    # Convert to Dataset format and add index
    dataset = Dataset.from_list(unique_events)
    dataset = add_idx_column(dataset)
    
    logger.info(f"Data split: {DATA_SPLIT}")
    logger.info(f"Dataset name: {dataset_name}")
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
    
    # Create output filename
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
        max_new_tokens=new_tokens, 
        num_generations=args.num_generations
    )
    
    # Save results as JSONL
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(all_results)} question results to {output_file}")
    
    # Log some statistics
    total_generations = len(all_results) * args.num_generations
    all_final_answers = []
    valid_count = 0
    
    for result in all_results:
        for final_answer in result['extracted_answer']:
            all_final_answers.append(final_answer)
            if final_answer is not None:
                valid_count += 1
    
    logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
    
    # Calculate accuracy for binary questions
    correct_count = 0
    for result in all_results:
        expected_label = result['resolution']  # 1 for Yes, 0 for No/None of the above
        for final_answer in result['extracted_answer']:
            if final_answer is not None:
                # Extract the answer key (YES/NO) and probability
                answer_key = list(final_answer.keys())[0] if final_answer else None
                if answer_key:
                    predicted_label = 1 if answer_key.strip().upper() == "YES" else 0
                    if predicted_label == expected_label:
                        correct_count += 1

    accuracy = correct_count / total_generations if total_generations > 0 else 0
    logger.info(f"Accuracy: {correct_count}/{total_generations} ({accuracy*100:.1f}%)")
    
    # Calculate statistics for numeric answers
    numeric_answers = []
    for answer in all_final_answers:
        if answer is not None:
            try:
                # Extract probability from the answer dict
                probability = list(answer.values())[0]
                if probability is not None:
                    numeric_val = float(probability)
                    numeric_answers.append(numeric_val)
            except (ValueError, TypeError, IndexError):
                pass

    if numeric_answers:
        logger.info(f"Numeric answers: {len(numeric_answers)}/{valid_count}")
        logger.info(f"Mean prediction: {np.mean(numeric_answers):.3f} ± {np.std(numeric_answers):.3f}")
        logger.info(f"Prediction range: [{np.min(numeric_answers):.3f}, {np.max(numeric_answers):.3f}]")
        
        # Calculate correlation with true labels if available
        true_labels = []
        predicted_values = []
        for result in all_results:
            if result.get('resolution') is not None:
                true_labels.append(result['resolution'])
                if result['extracted_answer'] and result['extracted_answer'][0] is not None:
                    try:
                        probability = list(result['extracted_answer'][0].values())[0]
                        if probability is not None:
                            predicted_values.append(float(probability))
                    except (ValueError, TypeError, IndexError):
                        pass

        if len(true_labels) == len(predicted_values) and len(true_labels) > 1:
            correlation = np.corrcoef(true_labels, predicted_values)[0, 1]
            logger.info(f"Correlation with true labels: {correlation:.3f}")
