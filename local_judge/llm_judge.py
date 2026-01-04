#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from functools import partial
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimal_tensor_parallel_size(available_gpus: int) -> int:
    """
    Get the optimal tensor parallel size based on available GPUs.
    Returns the largest power of 2 that is <= available_gpus.
    
    Args:
        available_gpus: Number of available GPUs
        
    Returns:
        Optimal tensor parallel size (1, 2, 4, or 8)
    """
    valid_sizes = [1, 2, 4, 8]
    for size in reversed(valid_sizes):
        if size <= available_gpus:
            return size
    return 1

# Define the prompt templates for the judge
def get_judge_prompt_with_gt(question, target, response, incorrect_options=None, cot=True):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        question: The question being asked
        target: The ground truth answer
        response: The response to judge
        incorrect_options: Optional string containing incorrect options
        cot: Whether to use a COT prompt
        
    Returns:
        A formatted prompt string for the judge
    """
    # The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.

    prompt = f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have the same information as in the ground-truth (not less nor unnecessary extra). 
The response can be more specific than the ground-truth (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be <= 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
"""

    if incorrect_options:
        prompt += f"\n{incorrect_options}"
        
    prompt += f"""Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."
        
# Think step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS.
# YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.

    return prompt

# Define the prompt templates for the judge
def prompt_without_tags(question, target, response, incorrect_options=None):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        question: The question being asked
        target: The ground truth answer
        response: The response to judge
        incorrect_options: Optional string containing incorrect options
        
    Returns:
        A formatted prompt string for the judge
    """
    prompt = f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have at least as much information as the ground-truth. 
The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible (correct)answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response.
This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" and nothing else. DO NOT JUSTIFY YOUR DECISION.
You must only output a single digit: 0 or 1."""
    return prompt


# Define the prompt templates for the judge
def get_continuous_judge_prompt_with_gt(question, target, response, incorrect_options=None, cot=True):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        question: The question being asked
        target: The ground truth answer
        response: The response to judge
        incorrect_options: Optional string containing incorrect options
        cot: Whether to use a COT prompt
        
    Returns:
        A formatted prompt string for the judge
    """
    # The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.

    prompt = f"""Your task is to score the given response to a question on a scale of 0 to 1, where 0 means the response does not match the ground truth answer and 1 means the response matches the ground truth answer. You are provided with a question, its ground truth response, and the response you need to score.
For a response to "match", it must have at least as much information as the ground-truth if the ground truth is not numeric or date related. 
The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
For numeric or date related answers, first compute the relative error, defined as |response - ground truth| / mean(response, ground truth). Then, the score of the response is 1 - relative error. For example, if the ground truth is 6 and the response is 4, then the relative error is |4 - 6| / mean(4, 6) = 2/5 = 0.4. Hence, the score of the response is 1 - 0.4 = 0.6.

Question: "{question}"
Ground truth: "{target}"
"""

    prompt += f"""Response: "{response}"

Your job is to SCORE the given response based on how close is to the ground truth answer in the context of the question. You should provide a continuous score only if the ground truth is numeric or date, otherwise provide only 0 or 1 (binary score). You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer in <answer> </answer> tags."""
    
    if cot:
        prompt += "\nThink step by step and end your response with <answer> XYZ </answer> TAGS where XYZ is the score (between 0 and 1)."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer> XYZ </answer> TAGS where XYZ is the score (between 0 and 1)."
        
# Think step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS.
# YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.

    return prompt



JUDGE_PROMPT_TEMPLATE_WITHOUT_GT = """
Your task is to judge whether the given response to a question is correct or not. You are only given a question and the response you are judging. 
The response should be correct if it has sufficient information to answer the question. It can have more information than necessary, and as long as that additional information is correct, the response should be judged as correct. If it is missing important information ASKED in the question, it should be judged as incorrect.
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as correct.

Possible judgments:
"0": The response is incorrect.
"1": The response is correct.
    
Question: "{question}"
Response: "{response}"
    
To the best of your knowledge: Does the provided response answer the question correctly? This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags. 
YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.
"""

# def load_existing_results(output_path: str) -> List[Dict[str, Any]]:
#     """
#     Load existing results from output file if it exists
    
#     Args:
#         output_path: Path to the output file
        
#     Returns:
#         List of dictionaries containing the results, or an empty list if no file exists
#     """
#     if os.path.exists(output_path):
#         logger.info(f"Loading existing results from {output_path}")
#         try:
#             data = []
#             with open(output_path, 'r') as f:
#                 for line in f:
#                     data.append(json.loads(line))
#             logger.info(f"Loaded {len(data)} existing results")
#             return data
#         except json.JSONDecodeError:
#             logger.warning(f"Failed to parse existing results from {output_path}, starting fresh")
#             return []
#     return []

def load_existing_results(data_path: str) -> List[Dict[str, Any]]:
    """
    Load existing results from input file
    
    Args:
        data_path: Path to the input file
        
    Returns:
        List of dictionaries containing the results
    """
    if os.path.exists(data_path):
        logger.info(f"Loading existing results from {data_path}")
        try:
            with open(data_path, 'r') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f if line.strip()]
                else:  # .json or other formats
                    data = json.load(f)
            logger.info(f"Loaded {len(data)} existing results")
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse existing results from {data_path}, starting fresh")
            return []
    return []

def save_results(data: List[Dict[str, Any]], output_path: str):
    """
    Save results to output file
    
    Args:
        data: List of dictionaries containing the results
        output_path: Path to the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # output_path = output_path.replace(".jsonl", "_judged.jsonl")
    
    # Save the results
    logger.info(f"Saving {len(data)} results to {output_path}")
    
    # Determine file format based on extension
    if output_path.endswith('.jsonl'):
        # Save as JSONL (one JSON object per line)
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        # Save as regular JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

def get_log_probs_vllm(model, tokenizer, prompts, token_ids_list, return_prob=False, gen_kwargs=None, max_tokens=2048, continuous=False):
    """
    Get log probabilities for specific token IDs using vLLM
    
    Args:
        model: vLLM model instance
        tokenizer: Tokenizer instance
        prompts: List of prompt strings
        token_ids_list: List of token IDs to get log probabilities for
        return_prob: Whether to return normalized probability in results (default: False)
        gen_kwargs: String of generation parameters (e.g. "temperature=0.7,top_p=0.9")
        max_tokens: Maximum number of tokens to generate (default: 2048)
        
    Returns:
        List of dictionaries containing generated text and optionally normalized probabilities
    """
    from vllm import SamplingParams
    import math
    
    # Default sampling parameters
    temperature = 0.0
    top_p = 0.95
    # top_k = -1
    min_p = 0.0
    do_sample = False
    n = 1  # Always generate 3 samples for each prompt by default
    
    # Parse gen_kwargs if provided
    if gen_kwargs:
        params_dict = {}
        for param in gen_kwargs.split(','):
            if '=' in param:
                key, value = param.split('=')
                key = key.strip()
                value = value.strip()
                
                if key == 'temperature':
                    temperature = float(value)
                elif key == 'top_p':
                    top_p = float(value)
                elif key == 'top_k':
                    top_k = int(value)
                elif key == 'min_p':
                    min_p = float(value)
                elif key == 'max_gen_toks':
                    max_tokens = int(value)
                elif key == 'n':
                    n = int(value)
                elif key == 'do_sample' and value.lower() == 'true':
                    do_sample = True
    
    # Use the parsed parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        logprobs=5,
        n=n
    )
    
    logger.info(f"Using generation parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, max_tokens={max_tokens}, do_sample={do_sample}")
    
    # Apply chat template to each prompt
    formatted_prompts = []
    for prompt in prompts:
        # Format the prompt using the model's chat template
        # Create a messages list with a single user message
        messages = [{"role": "user", "content": prompt}]
        try:
            # Apply the chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
            formatted_prompts.append(prompt)
    
    try:
        batch_outputs = model.generate(formatted_prompts, sampling_params)
    except Exception as e:
        logger.error(f"Failed to generate: {e}")
        return []
    
    logger.info(f"Number of prompts: {len(formatted_prompts)}")
    logger.info(f"Generated {len(batch_outputs)} outputs")
    results = []
    maxed_out = 0
    answer_tag_not_found = 0
    
    # Process each prompt/sample
    for i, sample_outputs in enumerate(batch_outputs):
        sample_results = []
        
        current_prompt = formatted_prompts[i]
        # Process each of the 3 generations for the current sample
        for j, output in enumerate(sample_outputs.outputs):
            # Extract logprobs from the vLLM output
            normalized_prob = None
            token_logprobs = {}
            
            # Store the full original response
            original_response = output.text if output.text else ""
            
            # Store completion tokens and prompt tokens
            completion_tokens = len(output.token_ids) if hasattr(output, 'token_ids') else None
            # Get prompt tokens from the sample_outputs (RequestOutput object)
            prompt_tokens = len(sample_outputs.prompt_token_ids) if hasattr(sample_outputs, 'prompt_token_ids') else None
            
            if completion_tokens and completion_tokens > 0:
                if completion_tokens >= max_tokens:
                    maxed_out += 1
                    
            # Check if we have logprobs in the output
            if hasattr(output, 'logprobs') and output.logprobs:
                logprobs_data = output.logprobs
                
                # Find the position of <answer> tag in the response
                answer_tag_position = -1
                answer_tag_found = False
                
                # First, find the token position of the <answer> tag
                for idx, logprob_entry in enumerate(logprobs_data):
                    for token, logprob_obj in logprob_entry.items():
                        if hasattr(logprob_obj, 'decoded_token') and '<answer>' in logprob_obj.decoded_token:
                            answer_tag_position = idx
                            answer_tag_found = True
                            break
                    if answer_tag_found:
                        break
                
                # Extract logprobs for tokens we care about (0 and 1) only after <answer> tag
                for token_id in token_ids_list:
                    # Find the token in the logprobs list, but only after the <answer> tag
                    for idx, logprob_entry in enumerate(logprobs_data):
                        if answer_tag_found and idx <= answer_tag_position:
                            continue  # Skip entries before or at the <answer> tag if it exists
                        
                        # Each entry is a dictionary mapping token_id to Logprob objects
                        for token, logprob_obj in logprob_entry.items():
                            # Check if this is one of our target tokens
                            if token == token_id or (hasattr(logprob_obj, 'decoded_token') and 
                                                    logprob_obj.decoded_token in ['0', '1']):
                                # Store using the decoded token as key
                                decoded_token = str(logprob_obj.decoded_token) if hasattr(logprob_obj, 'decoded_token') else str(token)
                                token_logprobs[decoded_token] = logprob_obj.logprob
                                # Once we find a 0 or 1 after the <answer> tag, we can stop looking
                                if answer_tag_found:
                                    break
                                    
                # Determine which token has the higher logprob and calculate its normalized probability
                if "1" in token_logprobs and "0" in token_logprobs:
                    # Convert log probabilities to probabilities
                    prob_1 = math.exp(token_logprobs["1"])
                    prob_0 = math.exp(token_logprobs["0"])
                    
                    # Determine which token has higher probability
                    if prob_1 > prob_0:
                        normalized_prob = prob_1 / (prob_1 + prob_0)
                        generation = "1"
                    else:
                        normalized_prob = prob_0 / (prob_1 + prob_0)
                        generation = "0"
                        
                elif "1" in token_logprobs:
                    normalized_prob = 1.0
                    generation = "1"
                elif "0" in token_logprobs:
                    normalized_prob = 1.0
                    generation = "0"
                else:
                    generation = original_response
                    
            elif original_response:
                generation = original_response
            else:
                generation = ""
                
            # Extract binary judgment (0 or 1) from response
            import re
            answer_matches = list(re.finditer(r'<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>', original_response))
            if answer_matches:
                # Get the last occurrence
                binary_judgment = answer_matches[0].group(1)
                generation = float(binary_judgment)
            else:
                if j < (n - 1):
                    continue # Skip this generation if <answer> tag not found but if it is the last generation, we should still return the generation
                
                answer_tag_not_found += 1
                
            # Round normalized probability to 3 decimal places if it's not None
            if normalized_prob is not None:
                normalized_prob = round(normalized_prob, 3)
                
            # Create result dictionary
            result = {
                'generation': generation,
                'full_response': original_response
            }
            
            # Add completion tokens and prompt tokens to the result
            if completion_tokens is not None:
                result['completion_tokens'] = completion_tokens
            if prompt_tokens is not None:
                result['prompt_tokens'] = prompt_tokens
            
            # Truncate full_response to only keep content up to the first </answer> tag
            if '</answer>' in original_response:
                result['full_response'] = original_response.split('</answer>')[0] + '</answer>'
            
            # Only include probability if return_prob is True
            if return_prob and normalized_prob is not None:
                result['normalized_prob'] = normalized_prob
                
            sample_results.append(result)
            
            if answer_matches:
                break 
            
            # if j == 0:
            #     print(f"Current prompt: {current_prompt}")
            #     print(f"Result: {result}")
            #     print("--------------------------------\n\n")
            
        if True:
            results.append(sample_results[0])
            continue
        
        # Count the number of 0s and 1s in the generations
        count_0 = 0
        count_1 = 0
        results_0 = []
        results_1 = []
        
        for result in sample_results:
            generation = result.get('generation', '').strip()
            binary_judgment = 1 if generation == "1" else 0
            
            if binary_judgment == 1:
                count_1 += 1
                results_1.append(result)
            else:
                count_0 += 1
                results_0.append(result)
        
        # Determine which judgment is the majority
        if count_1 > count_0:
            # 1 is the majority, pick the first result with judgment 1
            for i, result in enumerate(results_1):
                generation = result.get('generation', '').strip()
                if generation == "1":
                    best_result = result.copy()
                    break
                
            if not 'normalized_prob' in best_result and return_prob:
                best_result['normalized_prob'] = best_result.get('normalized_prob', 1.0)
        else:
            # 0 is the majority (or tie), pick the first result with judgment 0
            for i, result in enumerate(results_0):
                generation = result.get('generation', '').strip()
                if generation == "0":
                    best_result = result.copy()
                    break
                
            if not 'normalized_prob' in best_result and return_prob:
                best_result['normalized_prob'] = best_result.get('normalized_prob', 1.0)
        
        results.append(best_result)
        
    logger.info(f"Maxed out on {maxed_out} samples")
    # logger.info(f"Maxed out Percentage: {maxed_out / len(batch_outputs) * 100:.2f}%")
    logger.info(f"Answer tag not found Percentage: {answer_tag_not_found / len(batch_outputs) * 100:.2f}%")
    return results

def judge_responses_with_gt(
    data_path: str,
    judge_model_path: str,
    output_dir: str,
    use_token_logprobs: bool = True,
    batch_size: int = 32,
    n_gpus: int = 1,
    gen_kwargs: str = None,
    max_tokens: int = 2048,
    thinking: bool = False,
    continuous: bool = False,
    model: object = None,
    tokenizer: object = None
):
    """
    Judge model responses against ground truth and calculate normalized probabilities
    
    Args:
        data_path: Path to the JSON file containing model responses and ground truth
        judge_model_path: Path or HF identifier of the judge model
        output_dir: Directory to save the judgments
        use_token_logprobs: Whether to store normalized probabilities (ignored, now always stored)
        batch_size: Number of samples to process in a batch (only used for HF)
        gen_kwargs: Generation parameters like temperature, top_p, etc. (format: "temperature=0.7,top_p=0.9")
        max_tokens: Maximum number of tokens to generate
        model: Preloaded vLLM model instance (optional)
        tokenizer: Preloaded tokenizer instance (optional)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Load the input data
    logger.info(f"Loading data from {data_path}")
    # Handle different file formats
    data = load_existing_results(data_path)
    
    # data = data[:20]
    logger.info(f"Loaded input data with {len(data)} samples")
    
    # Use a shortened model name for the score field
    model_name = os.path.basename(judge_model_path.rstrip("/"))
    short_model_name = model_name.replace("-", "_").replace(".", "_")
    
    
    if continuous:
        template = partial(get_continuous_judge_prompt_with_gt, cot=True)
    else:
        template = partial(get_judge_prompt_with_gt, cot=True)
    # template = partial(get_judge_prompt_with_gt, cot=False)
    
    if "llama-2-70b" in judge_model_path:
        template = partial(get_judge_prompt_with_gt, cot=False)
        logger.info("Using non-COT prompt for Llama-2-70b")
        model_name = "llama-2-70b-chat-hf"
        short_model_name = "llama-2-70b"
        
    # Use the same input file for output
    output_path = data_path
    if continuous:
        score_field = f"continuous_score_{short_model_name}"
        prob_field = f"continuous_prob_{short_model_name}"
        response_field = f"continuous_response_{short_model_name}"
        tokens_field = f"continuous_completion_tokens_{short_model_name}"
        prompt_tokens_field = f"continuous_prompt_tokens_{short_model_name}"
    else:
        score_field = f"score_{short_model_name}"

        prob_field = f"prob_{short_model_name}"
        response_field = f"response_{short_model_name}"
        tokens_field = f"completion_tokens_{short_model_name}"
        prompt_tokens_field = f"prompt_tokens_{short_model_name}"
    
    
    # Load existing results if available - handle both freeform and standard formats
    existing_data = data
    # Create mapping using either existing question_id field or idx field (for freeform data)
    existing_map = {}
    for item in existing_data:
        # Use idx as question_id for freeform data, or existing question_id for standard data
        q_id = item.get("question_id", item.get("idx"))
        if q_id is not None:
            existing_map[q_id] = item
    
    # Identify which samples need judgment
    samples_to_judge = []
    sample_question_ids = []
    
    for sample in data:
        # Use idx as question_id for freeform data, or existing question_id for standard data
        question_id = sample.get("question_id", sample.get("idx"))
        
        # Skip if already judged
        if question_id in existing_map and score_field in existing_map[question_id]:
            continue
        
        # Skip if no extracted_answer
        if "extracted_answer" not in sample or len(sample["extracted_answer"]) == 0:
            continue
            
        samples_to_judge.append(sample)
        sample_question_ids.append(question_id)
    
    if not samples_to_judge:
        logger.info(f"All samples have already been judged by {model_name}, nothing to do")
        
        # Count existing entries for statistics
        correct_count = 0
        total_judged = sum(1 for item in existing_data if score_field in item)
        
        for item in existing_data:
            if score_field not in item:
                continue
                
            scores = item.get(score_field, [])
            if not continuous:
                # Binary scoring
                if isinstance(scores, list):
                    if len(scores) > 0 and isinstance(scores[0], dict):
                        # Dictionary format
                        if any(any(score == 1 for score in score_dict.values()) for score_dict in scores if isinstance(score_dict, dict)):
                            correct_count += 1
                    else:
                        # String format
                        if any(score == 1 for score in scores):
                            correct_count += 1
                elif scores == 1:
                    correct_count += 1
            else:
                # Continuous scoring
                if isinstance(scores, list):
                    if len(scores) > 0 and isinstance(scores[0], dict):
                        # Dictionary format - take max across all values
                        max_score = max(max(score_dict.values()) for score_dict in scores if isinstance(score_dict, dict))
                        correct_count += max_score
                    else:
                        # String format
                        correct_count += max(scores)
                else:
                    correct_count += scores
        
        if total_judged > 0:
            accuracy_percentage = correct_count / total_judged * 100
            logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
        else:
            logger.info(f"Summary for {model_name}: 0/0 correct (0.00%)")
    
        return
    
    logger.info(f"Need to judge {len(samples_to_judge)} out of {len(data)} samples with {model_name}")
    
    if len(samples_to_judge) < 3:
        for sample in samples_to_judge:
            logger.info(f"Sample {sample['idx']}: {sample['question_title']}")
            logger.info(f"Target: {sample['answer']}")
            logger.info(f"Response: {sample['extracted_answer']}")
            logger.info("--------------------------------")
    
    # Initialize the model if not provided
    if model is None or tokenizer is None:
        from vllm import LLM
        from transformers import AutoTokenizer
        
        logger.info(f"Initializing {model_name} judge via vLLM with {n_gpus} GPUs")
        
        # Ensure max_model_len is at least as large as max_tokens + context size
        # A reasonable buffer is to multiply max_tokens by 2
        max_model_len = max(4096, max_tokens * 2)
        max_model_len = 4096 # 8192 * 2
        
        model = LLM(
            model=judge_model_path,
            tensor_parallel_size=n_gpus,
            max_model_len=max_model_len,
            dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
    
    # Get token IDs for "0", "1"
    token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
    token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
    tokens_to_check = [token_0, token_1]
    
    # Prepare prompts for the judge
    prompts = []
    # template = prompt_without_tags
        
    # Prepare prompts - handle multiple extracted answers
    sample_indices = []  # Track which sample each prompt belongs to
    answer_indices = []  # Track which answer index each prompt corresponds to
    generation_indices = []  # Track which generation each prompt belongs to (for dict format)
    key_indices = []  # Track which key within generation each prompt corresponds to (for dict format)
    sample_dict_formats = []  # Track whether each sample uses dict format
    
    for i, sample in enumerate(samples_to_judge):
        # Handle both freeform and standard data formats
        question = sample.get("question", sample.get("question_title", ""))
        target = sample.get("target", sample.get("answer", ""))  # Ground truth answer
        
        # Handle response extraction for freeform vs standard formats
        responses_to_judge = []
        is_dict_format = False  # Track if we're dealing with dictionary format
        dict_structure = []     # Track the original dictionary structure for reconstruction
        
        if "extracted_answer" in sample:
            # Freeform format: use all extracted_answers
            extracted_answers = sample.get("extracted_answer", [])
            if extracted_answers:
                # Check if this is the new dictionary format
                if isinstance(extracted_answers[0], dict):
                    is_dict_format = True
                    # Extract all answer options from dictionaries
                    for gen_idx, answer_dict in enumerate(extracted_answers):
                        dict_keys = list(answer_dict.keys())
                        dict_structure.append(dict_keys)
                        for key in dict_keys:
                            responses_to_judge.append(key[-200:])  # Judge the key (answer option)
                else:
                    # Old string format
                    responses_to_judge = extracted_answers
            else:
                responses_to_judge = [""]
        elif "final_answer" in sample:
            extracted_answers = sample.get("final_answer", [])
            if extracted_answers:
                # Check if this is the new dictionary format
                if isinstance(extracted_answers[0], dict):
                    is_dict_format = True
                    # Extract all answer options from dictionaries
                    for gen_idx, answer_dict in enumerate(extracted_answers):
                        dict_keys = list(answer_dict.keys())
                        dict_structure.append(dict_keys)
                        for key in dict_keys:
                            responses_to_judge.append(key[-200:])  # Judge the key (answer option)
                else:
                    # Old string format
                    responses_to_judge = extracted_answers
            else:
                responses_to_judge = [""]
        else:
            # Standard format: use filtered_resps
            filtered_resps = sample.get("filtered_resps", "")
            if isinstance(filtered_resps, list):
                responses_to_judge = filtered_resps
            else:
                responses_to_judge = [filtered_resps]
        
        incorrect_options_text = None 
        
        # Generate incorrect options format
        if "options" in sample:
            options = sample.get("options", [])
            if len(options) > 0:
                answer_index = sample.get("answer_index", -1)
                incorrect_options_text = ""
                j = 0
                for qq, option in enumerate(options):
                    if qq != answer_index:  # Skip the correct option
                        incorrect_options_text += f"Incorrect option ({j+1}): \"{option}\"\n"
                        j += 1
        
        # Store the format type for this sample
        sample_dict_formats.append(is_dict_format)
        
        # Create prompts for each response
        if is_dict_format:
            # Dictionary format: track generation and key indices
            current_answer_idx = 0
            for gen_idx, dict_keys in enumerate(dict_structure):
                for key_idx, response in enumerate(dict_keys):
                    if "math" in data_path:
                        target_formatted = "$" + target + "$"
                        response_formatted = "$" + response + "$"
                    else:
                        target_formatted = target
                        response_formatted = response
                    
                    # Limit the response to 200 characters
                    response_formatted = response_formatted[-200:]
                    # Create the prompt for the judge
                    prompt = template(question=question, target=target_formatted, response=response_formatted)
                    if "qwen3" in short_model_name.lower():
                        if not thinking:
                            prompt += " /no_think"
                            
                    if i < 1 and current_answer_idx < 1:
                        logger.info(f"Sample {i}, Generation {gen_idx}, Key {key_idx}\nPrompt: {prompt}")
                            
                    prompts.append(prompt)
                    sample_indices.append(i)
                    answer_indices.append(current_answer_idx)
                    generation_indices.append(gen_idx)
                    key_indices.append(key_idx)
                    current_answer_idx += 1
        else:
            # String format: use original logic
            for answer_idx, response in enumerate(responses_to_judge):
                if "math" in data_path:
                    target_formatted = "$" + target + "$"
                    response_formatted = "$" + response + "$"
                else:
                    target_formatted = target
                    response_formatted = response
                
                # Limit the response to 200 characters
                response_formatted = response_formatted[-200:]
                
                # Create the prompt for the judge
                prompt = template(question=question, target=target_formatted, response=response_formatted)
                if "qwen3" in short_model_name.lower():
                    if not thinking:
                        prompt += " /no_think"
                        
                if i < 1 and answer_idx < 1:
                    logger.info(f"Sample {i}, Answer {answer_idx}\nPrompt: {prompt}")
                        
                prompts.append(prompt)
                sample_indices.append(i)
                answer_indices.append(answer_idx)
                generation_indices.append(-1)  # Not applicable for string format
                key_indices.append(-1)  # Not applicable for string format
    
    # Generate judgments and get normalized probabilities
    logger.info(f"Getting judgments and normalized probabilities for {len(prompts)} samples")
    
    # Get results with multiple generations handled internally by get_log_probs_vllm
    results = get_log_probs_vllm(model, tokenizer, prompts, tokens_to_check, return_prob=use_token_logprobs, gen_kwargs=gen_kwargs, max_tokens=max_tokens, continuous=continuous)
    
    # Process the judgments and update the existing data
    # Group results by sample index
    sample_results = {}
    for i, result in enumerate(results):
        sample_idx = sample_indices[i]
        answer_idx = answer_indices[i]
        
        if not continuous:
            # Extract binary judgment (0 or 1) from response
            # generation = result.get('generation', '').strip()
            # binary_judgment = 1 if generation == "1" else 0
            binary_judgment = float(result.get('generation', 0))
        else :
            binary_judgment = float(result.get('generation', 0))
        
        # Get full response text and normalized probability if available
        full_response = result.get('full_response', '')
        prob = result.get('normalized_prob')
        completion_tokens = result.get('completion_tokens')
        prompt_tokens = result.get('prompt_tokens')
        
        # Initialize sample results if not exists
        if sample_idx not in sample_results:
            sample_results[sample_idx] = {
                'judgments': [],
                'responses': [],
                'probs': [],
                'completion_tokens': [],
                'prompt_tokens': []
            }
        
        # Store results for this answer index (ensure correct order)
        while len(sample_results[sample_idx]['judgments']) <= answer_idx:
            sample_results[sample_idx]['judgments'].append(0)
            sample_results[sample_idx]['responses'].append("")
            sample_results[sample_idx]['probs'].append(None)
            sample_results[sample_idx]['completion_tokens'].append(None)
            sample_results[sample_idx]['prompt_tokens'].append(None)
        
        sample_results[sample_idx]['judgments'][answer_idx] = binary_judgment
        sample_results[sample_idx]['responses'][answer_idx] = full_response
        sample_results[sample_idx]['probs'][answer_idx] = prob
        sample_results[sample_idx]['completion_tokens'][answer_idx] = completion_tokens
        sample_results[sample_idx]['prompt_tokens'][answer_idx] = prompt_tokens
    
    # Update samples with aggregated results
    for sample_idx, results_data in sample_results.items():
        question_id = sample_question_ids[sample_idx]
        is_dict_format = sample_dict_formats[sample_idx]
        
        if is_dict_format:
            # Reconstruct dictionary format
            # Get the original structure for this sample
            sample = samples_to_judge[sample_idx]
            if "extracted_answer" in sample:
                original_structure = sample["extracted_answer"]
            elif "final_answer" in sample:
                original_structure = sample["final_answer"]
            else:
                original_structure = []
            
            # Reconstruct score dictionaries
            reconstructed_scores = []
            judgment_idx = 0
            
            for gen_idx, answer_dict in enumerate(original_structure):
                score_dict = {}
                for key in answer_dict.keys():
                    if judgment_idx < len(results_data['judgments']):
                        score_dict[key] = results_data['judgments'][judgment_idx]
                        judgment_idx += 1
                    else:
                        score_dict[key] = 0  # Default score if missing
                reconstructed_scores.append(score_dict)
            
            samples_to_judge[sample_idx][score_field] = reconstructed_scores
            
            # # Similarly reconstruct other fields if needed
            # if results_data['probs']:
            #     reconstructed_probs = []
            #     prob_idx = 0
            #     for gen_idx, answer_dict in enumerate(original_structure):
            #         prob_dict = {}
            #         for key in answer_dict.keys():
            #             if prob_idx < len(results_data['probs']):
            #                 prob_dict[key] = results_data['probs'][prob_idx]
            #                 prob_idx += 1
            #             else:
            #                 prob_dict[key] = None
            #         reconstructed_probs.append(prob_dict)
            #     samples_to_judge[sample_idx][prob_field] = reconstructed_probs
                
        else:
            # Original string format
            samples_to_judge[sample_idx][score_field] = results_data['judgments']
            
            # # Add token counts if available (filter out None values)
            # probs_list = [p for p in results_data['probs'] if p is not None]
            # if probs_list:
            #     samples_to_judge[sample_idx][prob_field] = probs_list
        
        # Add question_id field if it doesn't exist (for freeform data compatibility)
        if "question_id" not in samples_to_judge[sample_idx]:
            samples_to_judge[sample_idx]["question_id"] = question_id
        
        # # Add token counts if available (filter out None values)
        # completion_tokens_list = [ct for ct in results_data['completion_tokens'] if ct is not None]
        # prompt_tokens_list = [pt for pt in results_data['prompt_tokens'] if pt is not None]
        
        # if completion_tokens_list:
        #     samples_to_judge[sample_idx][tokens_field] = completion_tokens_list
        # if prompt_tokens_list:
        #     samples_to_judge[sample_idx][prompt_tokens_field] = prompt_tokens_list
    
    # Update the original data with judged samples and save back
    # Create a map of question_id to updated sample for easy lookup
    judged_samples_map = {}
    for sample in samples_to_judge:
        q_id = sample.get("question_id", sample.get("idx"))
        judged_samples_map[q_id] = sample
    
    # Update the original data with judged samples
    for i, sample in enumerate(existing_data):
        q_id = sample.get("question_id", sample.get("idx"))
        if q_id in judged_samples_map:
            existing_data[i] = judged_samples_map[q_id]
    
    # Save the complete updated dataset back to the original file
    save_results(existing_data, output_path)
    
    # Calculate and print summary statistics
    # Handle both string list and dictionary list formats
    total_judgments = 0
    correct_judgments = 0
    samples_with_correct_answers = 0
    
    for idx, item in enumerate(samples_to_judge):
        scores = item.get(score_field, [])
        is_dict_format = sample_dict_formats[idx] if idx < len(sample_dict_formats) else False
        
        if is_dict_format:
            # Dictionary format: scores is a list of dictionaries
            if isinstance(scores, list):
                for score_dict in scores:
                    if isinstance(score_dict, dict):
                        for key, score in score_dict.items():
                            total_judgments += 1
                            if score == 1:
                                correct_judgments += 1
                # Sample is correct if any judgment in any dictionary is correct
                sample_has_correct = any(
                    any(score == 1 for score in score_dict.values()) 
                    for score_dict in scores 
                    if isinstance(score_dict, dict)
                )
                if sample_has_correct:
                    samples_with_correct_answers += 1
        else:
            # Original string format
            if isinstance(scores, list):
                total_judgments += len(scores)
                correct_judgments += sum(scores)
                # Sample is considered correct if at least one judgment is correct
                if any(score == 1 for score in scores):
                    samples_with_correct_answers += 1
            elif scores == 1:
                total_judgments += 1
                correct_judgments += 1
                samples_with_correct_answers += 1
            else:
                total_judgments += 1
    
    total_samples = len(samples_to_judge)
    
    if total_judgments > 0:
        judgment_accuracy = correct_judgments / total_judgments * 100
        logger.info(f"Summary for {model_name}: {correct_judgments}/{total_judgments} individual judgments correct ({judgment_accuracy:.2f}%)")
    
    if total_samples > 0:
        sample_accuracy = samples_with_correct_answers / total_samples * 100
        logger.info(f"Summary for {model_name}: {samples_with_correct_answers}/{total_samples} samples with at least one correct answer ({sample_accuracy:.2f}%)")
    
    if total_judgments == 0 and total_samples == 0:
        logger.info(f"Summary for {model_name}: 0/0 correct (0.00%)")

def main():
    parser = argparse.ArgumentParser(description="Judge model responses and calculate normalized probabilities")
    parser.add_argument("--model_dir", type=str, required=True,
                      help="Path to the judge model directory (required)")
    parser.add_argument("--input_dir", 
                       default="/fast/nchandak/forecasting/evals/freeform/manual/theguardian_207/", 
                      help="Path to the directory containing samples.jsonl OR path to a specific file to judge")
    parser.add_argument("--input_file", type=str, default="",
                      help="Path to a specific file to judge (alternative to input_dir)")
    parser.add_argument("--output_dir", default="",
                      help="Directory to save the judgments (defaults to input_dir)")
    parser.add_argument("--logprobs", action="store_true",
                      help="Store normalized probabilities")
    parser.add_argument("--no_ground_truth", action="store_true",
                      help="Judge without using ground truth answers (judge purely from knowledge)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Number of samples to process in a batch")
    parser.add_argument('--thinking', action='store_true', 
                      help='Whether to use thinking mode generation parameters')
    parser.add_argument('--continuous', action='store_true', 
                      help='Whether to use continuous scoring')
    parser.add_argument('--gen_kwargs', type=str, default=None, 
                      help='Generation parameters like temperature, top_p, etc. (format: "temperature=0.7,top_p=0.9")')
    parser.add_argument('--max_tokens', type=int, default=2048, 
                      help='Maximum number of tokens to generate')
    
    args = parser.parse_args()
    
    # Validate model directory exists
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory {args.model_dir} does not exist")
        sys.exit(1)
    
    # Set generation parameters based on thinking mode
    if args.gen_kwargs is None:
        if args.thinking:
            # thinking mode parameters
            args.gen_kwargs = f"temperature=0.6,top_p=0.95,min_p=0,top_k=20,max_gen_toks={args.max_tokens},do_sample=true"
        else:
            # non-thinking mode parameters
            args.gen_kwargs = f"temperature=0.7,top_p=0.8,min_p=0,top_k=20,max_gen_toks={args.max_tokens},do_sample=true"
    
    # Determine input files to process
    samples_files = []
    
    if args.input_file:
        # Process specific file
        if os.path.exists(args.input_file):
            samples_files.append(args.input_file)
            # If output dir is not specified, use the directory of the input file
            output_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_file)
        else:
            logger.error(f"Input file {args.input_file} does not exist")
            sys.exit(1)
    else:
        # Only append files in the given directory (not recursively)
        for file in os.listdir(args.input_dir):
            file_path = os.path.join(args.input_dir, file)
            if os.path.isfile(file_path):
                samples_files.append(file_path)
                
        # If output dir is not specified, use input dir
        output_dir = args.output_dir if args.output_dir else args.input_dir
    
    for x in samples_files:
        print(x)
        
    if not samples_files:
        logger.error(f"Could not find any files to process")
        sys.exit(1)
    
    # Load models once and reuse for all files
    from vllm import LLM
    from transformers import AutoTokenizer
    
    # Get optimal GPU configuration
    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = get_optimal_tensor_parallel_size(available_gpus)
    
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Using tensor parallel size: {tensor_parallel_size}")
    
    model_dir = args.model_dir
    model_name = os.path.basename(model_dir.rstrip("/"))
    
    # Check available GPU memory before loading
    try:
        if torch.cuda.is_available():
            for i in range(available_gpus):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                memory_free = memory_total - memory_reserved
                logger.info(f"GPU {i}: {memory_free:.2f}GB free, {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # Warn if any GPU has less than 10GB free
                if memory_free < 10.0:
                    logger.warning(f"GPU {i} has only {memory_free:.2f}GB free memory, which may not be sufficient")
    except Exception as e:
        logger.warning(f"Could not check GPU memory: {e}")
    
    logger.info(f"Loading model {model_name} from {model_dir} with {tensor_parallel_size} GPUs")
    
    # Ensure max_model_len is at least as large as max_tokens + context size
    max_model_len = max(4096, args.max_tokens * 2)
    max_model_len = 4096 
    
    try:
        model = LLM(
            model=model_dir,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        logger.info(f"Successfully loaded model {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        sys.exit(1)
    
    # Process each file found
    for input_file in samples_files:
        # Create corresponding output directory structure
        if args.input_file:
            # For specific files, output to same directory
            file_output_dir = output_dir
        else:
            # For directory scanning, maintain directory structure
            relative_path = os.path.relpath(os.path.dirname(input_file), args.input_dir)
            file_output_dir = os.path.join(output_dir, relative_path)
        
        os.makedirs(file_output_dir, exist_ok=True)
        
        logger.info(f"Running model {model_name} on {input_file}")
        
        # Run the appropriate judging function
        judge_responses_with_gt(
            data_path=input_file,
            judge_model_path=model_dir,
            output_dir=file_output_dir,
            use_token_logprobs=args.logprobs,
            batch_size=args.batch_size,
            n_gpus=tensor_parallel_size,
            gen_kwargs=args.gen_kwargs,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
            continuous=args.continuous,
            model=model,
            tokenizer=tokenizer
        )
        
        # Clear memory after processing each file
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"Cleared GPU cache after processing {input_file}")
        except Exception as e:
            logger.warning(f"Error clearing GPU cache: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Cleanup model after processing all files
    try:
        del model
        del tokenizer
        logger.info(f"Deleted model {model_name} from memory")
    except Exception as e:
        logger.warning(f"Error deleting model {model_name}: {e}")
    
    # Final GPU cache clear
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Final GPU cache clear")
    except Exception as e:
        logger.warning(f"Error clearing GPU cache: {e}")
    
    # Force final garbage collection
    import gc
    gc.collect()
    logger.info("Memory cleanup completed")

if __name__ == "__main__":
    main() 