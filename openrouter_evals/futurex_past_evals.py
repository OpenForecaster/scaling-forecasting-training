#!/usr/bin/env python3
import re
import json
import os
import sys
import logging
import asyncio
import argparse
import numpy as np
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from datasets import Dataset, load_dataset
import ast 

# Import the existing OpenRouter inference engine
sys.path.append('/home/nchandak/forecasting')
from qgen.inference.openrouter_inference import OpenRouterInference

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_answer(completion: str) -> Optional[str]:
    """Extract the final answer from the LLM's output."""
    # Check if completion is None or not a string
    if completion is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(completion, str):
        completion = str(completion)
    
    # Check if completion is empty after conversion
    if not completion.strip():
        return None
    
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
        matches = re.finditer(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
        matches_list = list(matches)
    except:
        return None
    
    if not matches_list:
        return None 
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text


def extract_probability(completion: str) -> Optional[float]:
    """Extract the probability from the LLM's output."""
    # Check if completion is None or not a string
    if completion is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(completion, str):
        completion = str(completion)
    
    # Check if completion is empty after conversion
    if not completion.strip():
        return None
    
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
        matches = re.finditer(r"<probability>(.*?)<\/probability>", completion, re.DOTALL)
        matches_list = list(matches)
    except:
        return 1
    
    if not matches_list:
        return 1
    
    # Get the last match
    last_match = matches_list[-1]
    probability_text = last_match.group(1).strip()

    # Try to parse probability as float
    try:
        probability = float(probability_text)
        return probability
    except (ValueError, TypeError):
        return 1


def extract_boxed_answer(completion: str) -> Optional[str]:
    """Extract answers from \\boxed{...} format used in FutureX-Past."""
    if completion is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(completion, str):
        completion = str(completion)
    
    # Check if completion is empty after conversion
    if not completion.strip():
        return None
    
    if "</think>" in completion:
        completion = completion.split("</think>")[1].strip()
    
    try:
        # Look for \\boxed{...} pattern
        matches = re.finditer(r"\\boxed\{([^}]+)\}", completion, re.DOTALL)
        matches_list = list(matches)
    except:
        return None
    
    if not matches_list:
        return None 
    
    # Get the last match
    last_match = matches_list[-1]
    answer_text = last_match.group(1).strip()
    
    return answer_text


def format_futurex_past_prompt(
    question: str,
    options: List[str],
    end_time: str = "",
) -> str:
    """Format the prompt for FutureX-Past dataset (multiple choice format)."""
    
    # Format options
    options_text = ""
    if options:
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            options_text += f"{letter}. {option}\n"
    
    prompt = f"""You are an agent that can predict future events. The event to be predicted: "{question}"

{f"End time: {end_time}" if end_time else ""}

{options_text if options_text else ""}

Think step by step about the information provided and reason about the most likely outcome. Put your final answer in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

IMPORTANT: Your final answer MUST end with this exact format: listing all plausible options you have identified, separated by commas, within the box. For example: \\boxed{{A}} for a single option or \\boxed{{B, C, D}} for multiple options. Do not use any other format. Do not refuse to make a prediction. Do not say "I cannot predict the future." You must make a clear prediction based on the best data currently available, using the box format specified above.

Your response SHOULD STRICTLY END with <answer> </answer> tags, <probability> </probability> tags, and the \\boxed{{}} format.
"""

    return prompt


def add_binary_suffix(prompt: str) -> str:
    """Add the binary suffix to the prompt."""
    return prompt + """
Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

def add_binary_suffix_with_retrieval(prompt: str, retrieved_news_articles_summaries: str) -> str:
    """Add the binary suffix with retrieval to the prompt."""
    return prompt + f"""

Relevant passages retrieved from News Articles:
{retrieved_news_articles_summaries}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

def add_freeform_suffix(prompt: str) -> str:
    """Add the freeform suffix to the prompt."""
    return prompt + """
Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = -1.25 whereas if the answer was correct, then your score would be - (1 - 0.5)^2 = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be one of the options provided (A, B, C, etc.) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""

def add_freeform_suffix_with_retrieval(prompt: str, retrieved_news_articles_summaries: str) -> str:
    """Add the freeform suffix with retrieval to the prompt."""
    return prompt + f"""

Relevant passages retrieved from News Articles:
{retrieved_news_articles_summaries}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = -1.25 whereas if the answer was correct, then your score would be - (1 - 0.5)^2 = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be one of the options provided (A, B, C, etc.) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags.
"""


def add_numeric_suffix(prompt: str) -> str:
    """Add the numeric suffix to the prompt."""
    return prompt + """
Think step by step about the information provided, reason about uncertainty and give your best guess for the final answer (value of the event asked to upto two decimal places) in <answer> </answer> tags. 

You will be rewarded based on how close your prediction is to the actual value (1 - relative error).

Your response SHOULD STRICTLY END with <answer> </answer> tags.
"""


def format_futurex_past_binary_prompt(
    question: str,
    end_time: str = "",
) -> str:
    """Format the prompt for FutureX-Past binary questions."""
    
    prompt = f"""You are an agent that can predict future events. The event to be predicted: "{question}"

{f"End time: {end_time}" if end_time else ""}

Think step by step about the information provided and reason about the most likely outcome. Put your final answer (Yes or No) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

IMPORTANT: Your final answer MUST end with this exact format: \\boxed{{Yes}} or \\boxed{{No}}. Do not use any other format. Do not refuse to make a prediction. Do not say "I cannot predict the future." You must make a clear prediction based on the best data currently available, using the box format specified above.

Your response SHOULD STRICTLY END with <answer> </answer> tags, <probability> </probability> tags, and the \\boxed{{}} format.
"""

    return prompt


def add_idx_column(dataset: Dataset) -> Dataset:
    """Adds an 'idx' column to the dataset, storing the original row index."""
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)


def load_existing_results(output_file: str) -> Dict[int, Dict]:
    """Load existing results from JSONL file if it exists."""
    existing_results = {}
    
    if os.path.exists(output_file):
        logger.info(f"Found existing results file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line.strip())
                        idx = result.get('idx')
                        if idx is not None:
                            existing_results[idx] = result
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(existing_results)} existing results")
    
    return existing_results


def save_results_incrementally(results: List[Dict], output_file: str):
    """Save results to JSONL file."""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


async def evaluate_model(
    model_name: str,
    dataset: List[dict],
    output_file: str,
    num_generations: int = 1,
    max_tokens: int = 8192,
    batch_size: int = 5,
    num_articles: int = 10,
):
    """Run inference using the existing OpenRouterInference engine with incremental saving."""
    
    # Load existing results
    existing_results = load_existing_results(output_file)
    
    # Initialize the inference engine
    inference_engine = OpenRouterInference(
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.6  # Will be adjusted automatically based on model
    )
    
    # Determine what needs to be processed
    missing_prompts = []
    missing_metadata = []
    
    for i, row in enumerate(dataset):
        question_idx = row["idx"]
        
        # Format the prompt - handle different field names for retrieval vs regular dataset
        options = row.get("answer", row.get("futurex_original_answer", []))
        level = row.get("level", row.get("futurex_level", 0))
        
        # Handle options format - might be a list or string representation
        if isinstance(options, str) and options.startswith('['):
            try:
                options = ast.literal_eval(options)
            except (ValueError, SyntaxError):
                options = []
        elif not isinstance(options, list):
            options = []
        
        if level == 1:
            # Check if it's a binary question (Yes/No format)
            is_binary = len(options) == 1 and (
                "yes" in options[0].lower() or "no" in options[0].lower()
            )
        else:
            is_binary = False
            
        if level == 4:
            if len(options) > 1 :
                continue 
                
            ground_truth = row.get("answer", "")
            if ground_truth.startswith('[') and ground_truth.endswith(']'):
                ground_truth = ast.literal_eval(ground_truth)
                ground_truth = ground_truth[0]
            
            if not isinstance(ground_truth, float) and not isinstance(ground_truth, int):
                continue 
                
            
        # Check if this question already has complete results
        if question_idx in existing_results:
            existing_result = existing_results[question_idx]
            existing_responses = existing_result.get("response", [])
            existing_answers = existing_result.get("extracted_answer", [])
            
            # Check if we have all required generations
            if len(existing_responses) >= num_generations and len(existing_answers) >= num_generations:
                # Check if responses are valid (not empty/None)
                valid_responses = sum(1 for resp in existing_responses if resp and resp.strip() and ("\\boxed" in resp or "<answer" in resp or "<probability" in resp))
                
                extracted_answers = existing_result.get("extracted_answer", [])
                num_extracted_answers = sum(1 for ans in extracted_answers if ans and len(ans) > 0)
                if valid_responses >= num_generations and num_extracted_answers >= num_generations:
                    continue  # Skip this question, it's already complete
        
        # This question needs processing - add all generations for it
        for gen_idx in range(num_generations):
            og_prompt = row.get("prompt", "")
            
            # find "important" in the prompt
            if "IMPORTANT" in og_prompt:
                og_prompt = og_prompt.split("IMPORTANT")[0] 
            
            # Handle retrieval if available
            relevant_docs = row.get("relevant_articles_sorted_by_docs", [])
            retrieved_news_articles_summaries = ""
            
            if relevant_docs:
                j = 1
                for doc in relevant_docs[:num_articles]:
                    if len(doc) >= 3 and isinstance(doc[2], dict):
                        article_data = doc[2]
                        article_title = article_data.get("title", "")
                        article_passage = article_data.get("relevant_passage", "")
                        
                        if article_title and article_passage:
                            retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\nRelevant Passage: {article_passage}\n\n"
                            j += 1
            
            # Generate prompt with or without retrieval
            if retrieved_news_articles_summaries:
                # Use retrieval versions
                if is_binary or not options:
                    prompt = add_binary_suffix_with_retrieval(og_prompt, retrieved_news_articles_summaries)
                elif level <= 1:
                    prompt = add_freeform_suffix_with_retrieval(og_prompt, retrieved_news_articles_summaries)
                elif level == 4:
                    prompt = add_numeric_suffix(og_prompt)  # No retrieval version for numeric yet
                else:
                    assert False, "Invalid level"
            else:
                # Use original versions without retrieval
                if is_binary or not options:
                    prompt = add_binary_suffix(og_prompt)
                elif level <= 1:
                    prompt = add_freeform_suffix(og_prompt)
                elif level == 4:
                    prompt = add_numeric_suffix(og_prompt)
                else:
                    assert False, "Invalid level"
            
            if i == 0 and gen_idx == 0:
                logger.info(f"Sample prompt: {prompt}")
            
            missing_prompts.append(prompt)
            missing_metadata.append((row, gen_idx))
    
    logger.info(f"Found {len(missing_prompts)} prompts to process (out of {len(dataset) * num_generations} total)")

    if not missing_prompts:
        logger.info("All results already exist, nothing to process")
        # Convert existing results to list format
        all_results = list(existing_results.values())
        return all_results
    
    # Process in batches
    question_results = {}
    
    # Initialize question_results with existing data
    for idx, existing_result in existing_results.items():
        if idx in question_results:
            continue
        
        # Find the corresponding row
        row = None
        for r in dataset:
            if r["idx"] == idx:
                row = r
                break
        
        if row:
            question_results[idx] = {
                "row": row,
                "responses": existing_result.get("response", []),
                "final_answers": existing_result.get("extracted_answer", []),
                "prompt_tokens": existing_result.get("prompt_tokens", []),
                "completion_tokens": existing_result.get("completion_tokens", []),
                "reasoning": existing_result.get("reasoning", []),
                "final_prompts": existing_result.get("final_prompts", []),
                "is_binary": existing_result.get("is_binary", []),
            }
    
    # Process missing prompts in batches
    for batch_start in tqdm(range(0, len(missing_prompts), batch_size), desc=f"Processing {model_name}"):
        batch_end = min(batch_start + batch_size, len(missing_prompts))
        batch_prompts = missing_prompts[batch_start:batch_end]
        batch_metadata = missing_metadata[batch_start:batch_end]
        
        # Generate completions for this batch
        batch_completions = await inference_engine.generate(
            prompts=batch_prompts,
            batch_size=batch_size
        )
        
        # Process batch results
        for (row, gen_idx), completion, final_prompt in zip(batch_metadata, batch_completions, batch_prompts):
            question_idx = row["idx"]
            
            if question_idx not in question_results:
                question_results[question_idx] = {
                    "row": row,
                    "responses": [],
                    "final_answers": [],
                    "prompt_tokens": [],
                    "completion_tokens": [],
                    "reasoning": [],
                    "final_prompts": [],
                    "is_binary": [],
                }
            
            response = None
            
            # Handle None completions (failed requests)
            if completion is None:
                completion = ""
                final_ans = {}
                prompt_tokens = 0
                completion_tokens = 0
                reasoning = ""
                final_prompt = ""
                is_binary = 0
            else:
                response = completion['response']
                prompt_tokens = completion['prompt_tokens']
                completion_tokens = completion['completion_tokens']
                reasoning = completion['reasoning']
                final_prompt = final_prompt
                is_binary = 0 
                
                if "that the event asked will resolve to YES" in final_prompt:
                    is_binary = 1
                
                # Extract answer and probability
                boxed_ans = extract_boxed_answer(response)
                regular_ans = extract_answer(response)
                final_prob = extract_probability(response)
                
                # Use boxed answer if available, otherwise use regular answer
                final_answer_text = boxed_ans if boxed_ans else regular_ans
                if final_prob is not None and not final_answer_text:
                    if final_prob > 0.5:
                        final_answer_text = "YES"
                    else:
                        final_answer_text = "NO"
                        final_prob = 1 - final_prob
                    
                final_ans = {final_answer_text: final_prob} if final_answer_text else {}
            
            # Ensure we have the right number of slots
            while len(question_results[question_idx]["responses"]) <= gen_idx:
                question_results[question_idx]["responses"].append("")
                question_results[question_idx]["final_answers"].append({})
                question_results[question_idx]["prompt_tokens"].append(0)
                question_results[question_idx]["completion_tokens"].append(0)
                question_results[question_idx]["reasoning"].append("")
                question_results[question_idx]["final_prompts"].append("")
                question_results[question_idx]["is_binary"].append("")
                
                
            # Store the result at the correct generation index
            question_results[question_idx]["responses"][gen_idx] = response
            question_results[question_idx]["final_answers"][gen_idx] = final_ans
            question_results[question_idx]["prompt_tokens"][gen_idx] = prompt_tokens
            question_results[question_idx]["completion_tokens"][gen_idx] = completion_tokens
            question_results[question_idx]["reasoning"][gen_idx] = reasoning
            question_results[question_idx]["final_prompts"][gen_idx] = final_prompt
            question_results[question_idx]["is_binary"][gen_idx] = is_binary
            
        # Save progress after each batch
        current_results = []
        for question_idx, data in question_results.items():
            row = data["row"]
            
            result = {
                "model": model_name,
                "split": "eval",
                "data_type": "futurex_past",
                "idx": question_idx,
                "response": data["responses"],
                "extracted_answer": data["final_answers"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "reasoning": data["reasoning"],
                # FutureX-Past specific fields
                "question_id": row.get("question_id", ""),
                "question": row.get("question", row.get("question_title")),
                "answer": row.get("answer", ""),
                "options": row.get("options", []),
                "end_time": row.get("end-time", ""),
                "level": row.get("level", 0),
                "original_prompt": row.get("prompt", ""),
                "final_prompt": data["final_prompts"],
                "is_binary": data["is_binary"],
            }
            
            current_results.append(result)
        
        # Save incremental results
        save_results_incrementally(current_results, output_file)
        logger.info(f"Saved progress: {len(current_results)} results to {output_file}")
        
        # Small delay between batches
        await asyncio.sleep(1)
    
    # Convert to final result format
    all_results = []
    for question_idx, data in question_results.items():
        row = data["row"]
        
        result = {
            "model": model_name,
            "split": "eval",
            "data_type": "futurex_past",
            "idx": question_idx,
            "response": data["responses"],
            "extracted_answer": data["final_answers"],
            "prompt_tokens": data["prompt_tokens"],
            "completion_tokens": data["completion_tokens"],
            "reasoning": data["reasoning"],
            # FutureX-Past specific fields
            "question_id": row.get("question_id", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "options": row.get("options", []),
            "end_time": row.get("end-time", ""),
            "level": row.get("level", 0),
            "original_prompt": row.get("prompt", ""),
            "final_prompt": data["final_prompts"],
            "is_binary": data["is_binary"],
        }
        
        all_results.append(result)
    
    return all_results


async def main():
    parser = argparse.ArgumentParser(description="FutureX-Past evaluation using OpenRouter API")
    parser.add_argument('--base_save_dir', default="/home/nchandak/forecasting/evals/futurex_past", 
                       help="Base directory to save outputs")
    parser.add_argument('--data_split', type=str, default="train", 
                       help="Data split to use (train/test/validation)")
    parser.add_argument('--num_generations', type=int, default=1, 
                       help="Number of generations to use per prompt")
    parser.add_argument('--max_tokens', type=int, default=32768, 
                       help="Maximum number of tokens for generation")
    parser.add_argument('--models', nargs='+', default=[None],
                       help="List of models to evaluate")
    parser.add_argument('--batch_size', type=int, default=400,
                       help="Batch size for API requests")
    parser.add_argument('--level_filter', type=int, nargs='+', default=[1],
                       help="Filter dataset to only include questions of these levels (default: [1])")
    parser.add_argument('--retrieval_dataset', type=str, default="", #/fast/nchandak/forecasting/datasets/futurex/with_retrieval/futurex-withretrieval_past_train_level_1_size_86_30.jsonl",
                       help="Path to retrieval dataset JSONL file (if not provided, uses HuggingFace dataset)")
    parser.add_argument('--num_articles', type=int, default=10,
                       help="Number of retrieved articles to include in prompt")
    
    args = parser.parse_args()
    
    # Create output directory structure
    output_base_dir = args.base_save_dir
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Output directory: {output_base_dir}")
    
    # Load dataset - either retrieval dataset or HuggingFace dataset
    if args.retrieval_dataset:
        logger.info(f"Loading FutureX-Past retrieval dataset from: {args.retrieval_dataset}")
        questions_data = []
        
        with open(args.retrieval_dataset, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        # Add idx if not present
                        if 'idx' not in item:
                            item['idx'] = line_idx
                        questions_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_idx}: {e}")
                        continue
        
        logger.info(f"Original retrieval dataset size: {len(questions_data)}")
        
        # Filter by level if specified
        if args.level_filter:
            filtered_items = []
            for item in questions_data:
                # Handle different level field names
                level = item.get('futurex_level', item.get('level', 0))
                if int(level) in args.level_filter:
                    filtered_items.append(item)
            questions_data = filtered_items
            logger.info(f"Filtered to level {args.level_filter}: {len(questions_data)} questions")
    else:
        logger.info(f"Loading FutureX-Past dataset from HuggingFace")
        dataset = load_dataset('futurex-ai/Futurex-Past', split=args.data_split)
        
        logger.info(f"Original dataset size: {len(dataset)}")
        
        # Filter to only specified levels
        filtered_items = []
        for item in dataset:
            if int(item['level']) in args.level_filter:
                filtered_items.append(item)
        logger.info(f"Filtered to level {args.level_filter}: {len(filtered_items)} questions")
        
        # Convert to Dataset format and add index
        dataset = Dataset.from_list(filtered_items)
        dataset = add_idx_column(dataset)
        
        # Convert to list format for processing
        questions_data = []
        for item in dataset:
            questions_data.append(dict(item))
    
    logger.info(f"Data split: {args.data_split}")
    logger.info(f"Dataset size: {len(questions_data)}")
    logger.info(f"Level filter: {args.level_filter}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Using retrieval: {args.retrieval_dataset is not None}")
    if args.retrieval_dataset:
        logger.info(f"Number of articles per prompt: {args.num_articles}")
    
    # Available models on OpenRouter for FutureX-Past evaluation
    models = [
        # "openai/gpt-4o",
        # "openai/o4-mini-high", 
        # "google/gemini-2.5-pro-preview",
        # "meta-llama/llama-3.3-70b-instruct",
        # "google/gemini-2.5-flash-preview",
        # "meta-llama/llama-4-maverick",
        "openai/gpt-oss-120b",
        # "x-ai/grok-3-mini",
        "qwen/qwen3-8b",
        # "deepseek/deepseek-chat-v3-0824",
        # "qwen/qwen-2.5-72b-instruct",
    ]
    
    # Handle models list
    if args.models == [None] or not args.models or args.models[0] is None:
        args.models = models
    
    logger.info(f"Models to evaluate: {args.models}")
    
    # Process each model
    for model_name in args.models:
        logger.info(f"Evaluating model: {model_name}")
        
        # Create output filename
        model_clean = model_name.split("/")[-1]
        retrieval_suffix = f"_retrieval_{args.num_articles}" if args.retrieval_dataset else ""
        output_file = os.path.join(
            output_base_dir, 
            f"{model_clean}_{args.data_split}_level_{','.join(map(str, args.level_filter))}_size_{len(questions_data)}_generations_{args.num_generations}{retrieval_suffix}.jsonl"
        )
        
        # Run evaluation
        all_results = await evaluate_model(
            model_name=model_name,
            dataset=questions_data,
            output_file=output_file,
            num_generations=args.num_generations,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            num_articles=args.num_articles,
        )
        
        # Final save (in case there were no new batches to process)
        save_results_incrementally(all_results, output_file)
        logger.info(f"Final save: {len(all_results)} question results to {output_file}")
        
        # Log some statistics
        total_generations = len(all_results) * args.num_generations
        valid_count = 0
        
        # Count valid answers
        for result in all_results:
            for final_answer in result['extracted_answer']:
                if final_answer is not None and len(final_answer) > 0:
                    valid_count += 1
        
        logger.info(f"Valid answers extracted: {valid_count}/{total_generations} ({valid_count/total_generations*100:.1f}%)")
        
        # Calculate accuracy for multiple choice questions
        correct_count = 0
        total_with_ground_truth = 0
        
        for result in all_results:
            ground_truth = result.get('answer', [])
            if ground_truth:  # Only evaluate if we have ground truth
                ground_truth = ground_truth[2:-2].lower()
                # print(f"Ground truth: {ground_truth}")
                total_with_ground_truth += len(result['extracted_answer'])
                for final_answer in result['extracted_answer']:
                    if final_answer is not None and len(final_answer) > 0:
                        # Extract the predicted answer
                        predicted = list(final_answer.keys())[0].lower() if final_answer else None
                        if predicted:
                            correctness = 0
                            # Check if prediction matches any ground truth answer
                            if isinstance(ground_truth, list):
                                if predicted.lower() in ground_truth or any(pred.lower().strip() in gt for gt in ground_truth for pred in [predicted]):
                                    correctness = 1
                            else:
                                if predicted.lower() == ground_truth or predicted.lower().strip() in str(ground_truth):
                                    correctness = 1
                                    
                            if correctness == 1:
                                correct_count += 1
                            else:
                                # print(f"Incorrect prediction: {predicted} for ground truth: {ground_truth}")
                                if len(ground_truth) > 1 and len(predicted) == 1:
                                    print(f"Incorrect prediction: {predicted} for ground truth: {ground_truth}")
                                    # print prompt 
                                    print(f"Prompt: {result['original_prompt']}\n")
                                    print(f"Response: {result['response']}\n")
                                    print(f"Options: {result['options']}\n")
                                    # print(f"Final answer: {result['extracted_answer']}")
                                    # print(f"Ground truth: {ground_truth}")
                                    # print(f"Predicted: {predicted}")
                                    # print(f"Correctness: {correctness}")
                                    print("--------------------------------")

        if total_with_ground_truth > 0:
            accuracy = correct_count / total_with_ground_truth
            logger.info(f"Accuracy: {correct_count}/{total_with_ground_truth} ({accuracy*100:.1f}%)")
        
        # Calculate statistics for probabilities
        numeric_answers = []
        for answer in [ans for result in all_results for ans in result['extracted_answer']]:
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
            logger.info(f"Mean confidence: {np.mean(numeric_answers):.3f} ± {np.std(numeric_answers):.3f}")
            logger.info(f"Confidence range: [{np.min(numeric_answers):.3f}, {np.max(numeric_answers):.3f}]")
        
        # Small delay between models
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main()) 