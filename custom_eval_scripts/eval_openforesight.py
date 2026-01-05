"""
Evaluation script for nikhilchandak/OpenForesight dataset.
Evaluates models on the test set using pre-formatted prompts from the dataset.
Uses vLLM for efficient inference.
"""

import json
import os
import sys
import time
import numpy as np
import torch
from typing import List, Dict
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Import common utilities
from utils import (
    setup_seeds, setup_logging, setup_environment,
    extract_answer, extract_probability,
    load_model_and_tokenizer, apply_chat_template
)

# Setup
setup_seeds()
setup_environment()
logger = setup_logging()

def load_openforesight_dataset(split: str = "test", max_samples: int = None) -> List[Dict]:
    """
    Load the OpenForesight dataset from Hugging Face.
    
    Args:
        split: Dataset split to load (train/validation/test)
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dictionaries containing dataset examples
    """
    logger.info(f"Loading nikhilchandak/OpenForesight dataset, split: {split}")
    
    try:
        dataset = load_dataset("nikhilchandak/OpenForesight", split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")
        
        # Convert to list of dictionaries
        data_list = []
        for i, item in enumerate(dataset):
            # Add an index field if not present
            item_dict = dict(item)
            if 'idx' not in item_dict:
                item_dict['idx'] = i
            data_list.append(item_dict)
        
        logger.info(f"Loaded {len(data_list)} examples from OpenForesight {split} split")
        return data_list
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def evaluate_model(
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    without_retrieval: bool = False,
):
    """
    Run batched inference with multiple generations per prompt using vLLM.
    Uses pre-formatted prompts from the dataset.
    """
    all_prompts = []
    all_idxs = []
    all_row_data = []
    
    # Determine which prompt field to use
    prompt_field = "prompt_without_retrieval" if without_retrieval else "prompt"
    
    for i, row in enumerate(dataset):
        # Use the pre-formatted prompt from the dataset
        if prompt_field not in row or not row[prompt_field]:
            logger.warning(f"Row {i} missing '{prompt_field}' field, skipping")
            continue
        
        local_prompt = row[prompt_field]
        
        # Apply chat template
        prompt = apply_chat_template(tokenizer, local_prompt, model_name)
        
        all_prompts.append(prompt)
        all_idxs.append(row.get("idx", i))
        all_row_data.append(row)
    
    logger.info(f"Prepared {len(all_prompts)} prompts for evaluation")
    
    # Configure sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=num_generations,
    )
    
    # Process all prompts with vLLM
    logger.info(f"Starting generation with vLLM for {len(all_prompts)} prompts, {num_generations} generations each")
    start_time = time.time()
    
    # Generate completions using vLLM's batched API
    all_outputs = model.generate(all_prompts, sampling_params)
    
    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Process results
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
            answer = generated_text
            
            # Calculate token counts
            completion_tokens = len(tokenizer.encode(answer))
            
            # Extract single answer
            last_ans = extract_answer(answer)
            final_prob = extract_probability(answer)
            
            if last_ans is None and final_prob:
                last_ans = "YES"
            
            final_ans = {last_ans: final_prob}
            
            responses.append(answer)
            completion_tokens_list.append(completion_tokens)
            final_answers.append(final_ans)
        
        # Calculate prompt tokens once per prompt
        prompt_tokens = len(tokenizer.encode(prompt))
        
        # Store result
        result = {
            "model": model_name,
            "idx": idx,
            "response": responses,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens_list,
            "extracted_answer": final_answers,
            "without_retrieval": without_retrieval,
            # Preserve original dataset fields
            "question_title": row.get("question_title", ""),
            "background": row.get("background", ""),
            "resolution_criteria": row.get("resolution_criteria", ""),
            "answer": row.get("answer", ""),
            "answer_type": row.get("answer_type", ""),
            "resolution_date": row.get("resolution_date", ""),
            "question_start_date": row.get("question_start_date", ""),
            "data_source": row.get("data_source", ""),
            "article_url": row.get("url", row.get("article_url", "")),
        }
        
        all_results.append(result)
    
    # Log mean output token length with standard deviation
    all_completion_tokens = []
    for result in all_results:
        all_completion_tokens.extend(result["completion_tokens"])
    
    if all_completion_tokens:
        mean_output_length = np.mean(all_completion_tokens)
        std_output_length = np.std(all_completion_tokens)
        logger.info(f"Mean output token length: {mean_output_length:.2f} ± {std_output_length:.2f}")
    
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models on nikhilchandak/OpenForesight dataset")
    parser.add_argument('--base_save_dir', default="/fast/nchandak/forecasting/evals/openforesight", 
                        help="Base directory to save outputs")
    
    parser.add_argument('--model_dir', type=str, default="/fast/nchandak/models/Qwen3-8B", 
                        help="Model directory")
    parser.add_argument('--model', type=str, default="None", help="Model name (overrides auto-detection)")
    
    parser.add_argument('--max_new_tokens', type=int, default=16384, 
                        help="Maximum number of new tokens for generation")
    
    parser.add_argument('--data_split', type=str, default="test", 
                        help="Data split to use (train/validation/test)")
    
    parser.add_argument('--num_generations', type=int, default=1, 
                        help="Number of generations per prompt")
    
    parser.add_argument('--without_retrieval', action='store_true', 
                        help="Use prompt_without_retrieval field instead of prompt field")
    
    parser.add_argument('--max_samples', type=int, default=None, 
                        help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_base_dir = args.base_save_dir
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {gpu_count}")
    
    # Load dataset
    dataset = load_openforesight_dataset(split=args.data_split, max_samples=args.max_samples)
    
    if not dataset:
        logger.error("No data loaded from dataset")
        sys.exit(1)
    
    logger.info(f"Dataset split: {args.data_split}")
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Determine model name
    model_name = args.model
    if args.model == "None":
        model_name = args.model_dir.rstrip("/").split("/")[-1]
        if "__" in model_name:
            model_name = model_name.split("__")[1]
    
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    
    # Create output filename
    retrieval_suffix = "_without_retrieval" if args.without_retrieval else ""
    output_file = os.path.join(
        output_base_dir,
        f"{model_name}_{args.data_split}_size_{len(dataset)}_generations_{args.num_generations}{retrieval_suffix}.jsonl"
    )
    logger.info(f"Output file: {output_file}")
    
    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Exiting without running evaluation.")
        sys.exit(0)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir, model_name)
    
    # Run evaluation
    all_results = evaluate_model(
        model_name,
        model,
        tokenizer,
        dataset,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        without_retrieval=args.without_retrieval,
    )
    
    # Save results as JSONL
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(all_results)} results to {output_file}")
    
    # Log statistics
    total_generations = len(all_results) * args.num_generations
    all_final_answers = []
    valid_count = 0
    
    # For single outcomes
    for result in all_results:
        for final_answer in result['extracted_answer']:
            all_final_answers.append(final_answer)
            if final_answer is not None:
                valid_count += 1
    
    logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")

