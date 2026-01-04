#!/usr/bin/env python3
import json
import os
import argparse
import logging
from datasets import Dataset, load_dataset
from typing import List, Dict, Any
import ast
from datetime import datetime

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def add_idx_column(dataset: Dataset) -> Dataset:
    """Adds an 'idx' column to the dataset, storing the original row index."""
    return dataset.map(lambda example, idx: {'idx': idx}, with_indices=True)



def add_binary_suffix(prompt: str) -> str:
    """Add the binary suffix to the prompt."""
    return prompt + """
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


def determine_answer_type(options: List[str], level: int) -> str:
    """Determine the answer type based on options and level."""
    
    if level == 1:
        if len(options) == 1 and ("yes" in options[0].lower() or "no" in options[0].lower()):
            return "binary (Yes/No)"
        else:
            return "multiple choice"
    else:
        return ""


def extract_ground_truth_answer(options: List[str], level: int) -> str:
    """Extract the ground truth answer from options."""
    
    if not options:
        return ""
    
    if level == 1:
        if len(options) == 1 and ("yes" in options[0].lower() or "no" in options[0].lower()):
            # Binary question - return Yes or No
            return "Yes" if "yes" in options[0].lower() else "No"
        else:
            # Multiple choice - return the first option as placeholder
            return options[0]
    else:
        # For other levels, return the first option or empty string
        return options[0] if options else ""


def add_prefix(prompt: str) -> str:
    """Add the prefix to the prompt."""
    return f"You will be asked a forecasting question. You have to come up with the best guess for the final answer.\nQuestion: {prompt}"

def save_dataset_to_jsonl(dataset_items: List[Dict], output_file: str):
    """Save dataset items to JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in dataset_items:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved {len(dataset_items)} items to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Save FutureX-Past dataset in JSONL format")
    parser.add_argument('--output_dir', default="/fast/nchandak/forecasting/datasets/futurex/", 
                       help="Output directory to save the dataset")
    parser.add_argument('--data_split', type=str, default="train", 
                       help="Data split to use (train/test/validation)")
    parser.add_argument('--level_filter', type=int, nargs='+', default=[1],
                       help="Filter dataset to only include questions of these levels (default: all levels)")
    parser.add_argument('--max_items', type=int, default=None,
                       help="Maximum number of items to save (default: all)")
    parser.add_argument('--dataset_name', type=str, default="futurex_past",
                       help="Name for the dataset in output filename")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load FutureX-Past dataset from HuggingFace
    logger.info(f"Loading FutureX-Past dataset from HuggingFace")
    dataset = load_dataset('futurex-ai/Futurex-Past', split=args.data_split)
    
    logger.info(f"Original dataset size: {len(dataset)}")
    
    # Filter to only specified levels if provided
    filtered_items = []
    for item in dataset:
        if args.level_filter is None or int(item['level']) in args.level_filter:
            filtered_items.append(item)
    
    if args.level_filter:
        logger.info(f"Filtered to level {args.level_filter}: {len(filtered_items)} questions")
    else:
        logger.info(f"Using all levels: {len(filtered_items)} questions")
    
    # Convert to Dataset format and add index
    dataset = Dataset.from_list(filtered_items)
    dataset = add_idx_column(dataset)
    
    # Limit items if specified
    if args.max_items and len(dataset) > args.max_items:
        dataset = dataset.select(range(args.max_items))
        logger.info(f"Limited to {args.max_items} items")
    
    # Process dataset items
    processed_items = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for item in dataset:
        # Get basic information
        question_id = item.get("question_id", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        end_time = item.get("end-time", "")
        level = int(item.get("level", 0))
        idx = item.get("idx", 0)
        prompt = item.get("prompt", "")
        
        # Parse options from answer field (it's stored as string representation of list)
        try:
            options = ast.literal_eval(answer) if answer else []
        except (ValueError, SyntaxError):
            options = []
        
        # Determine if it's a binary question
        is_binary = False
        if level == 1 and len(options) == 1:
            is_binary = "yes" in options[0].lower() or "no" in options[0].lower()
            is_binary = 1 if is_binary else 0
            
        
        og_prompt = prompt
        
        # find "important" in the prompt
        if "IMPORTANT" in og_prompt:
            og_prompt = og_prompt.split("IMPORTANT")[0] 
            text = "You are an agent that can predict future events."
            pos = og_prompt.find(text)
            if pos != -1:
                og_prompt = og_prompt[pos + len(text):]
                og_prompt = add_prefix(og_prompt)
        
        if is_binary or not options:
            prompt = add_binary_suffix(og_prompt)
        elif level <= 1:
            prompt = add_freeform_suffix(og_prompt)
        else:
            assert False, "Invalid level"
        
        # Determine answer type
        answer_type = determine_answer_type(options, level)
        
        # Extract ground truth answer
        ground_truth_answer = extract_ground_truth_answer(options, level)
        
        # Create background information
        # background = f"This is a forecasting question from the FutureX-Past dataset (Level {level})."
        # if end_time:
        #     background += f" The resolution date is {end_time}."
        background = ""
        
        # Create processed item in the target format
        processed_item = {
            "question_title": question,
            "background": background,
            "resolution_criteria": "",
            "answer_type": answer_type,
            "answer": ground_truth_answer,
            "url": question_id,  # FutureX-Past doesn't have URLs
            "data_source": "futurex_past",
            "original_file": f"futurex-ai/Futurex-Past_{args.data_split}",
            "resolution_date": end_time,
            "question_close_date": end_time,
            "question_start_date": "",
            # Additional FutureX-Past specific fields
            "futurex_question_id": question_id,
            "question_id": question_id,
            "futurex_level": level,
            "futurex_options": item.get("options", ""),
            "is_binary": is_binary,
            "futurex_original_answer": answer,
            "idx": idx,
            "futurex_prompt": item.get("prompt", ""),
            "prompt": prompt,
        }
        
        processed_items.append(processed_item)
    
    logger.info(f"Processed {len(processed_items)} items")
    
    # Create output filename
    level_str = f"level_{','.join(map(str, args.level_filter))}" if args.level_filter else "all_levels"
    size_str = f"size_{len(processed_items)}"
    output_filename = f"{args.dataset_name}_{args.data_split}_{level_str}_{size_str}_cleaned.jsonl"
    output_file = os.path.join(args.output_dir, output_filename)
    
    # Save to JSONL file
    save_dataset_to_jsonl(processed_items, output_file)
    
    # Log some statistics
    logger.info(f"Dataset statistics:")
    logger.info(f"  Split: {args.data_split}")
    logger.info(f"  Total items: {len(processed_items)}")
    logger.info(f"  Levels: {set(item['futurex_level'] for item in processed_items)}")
    logger.info(f"  Binary questions: {sum(1 for item in processed_items if item['futurex_is_binary'])}")
    logger.info(f"  Multiple choice questions: {sum(1 for item in processed_items if not item['futurex_is_binary'])}")
    
    # Count questions by level
    level_counts = {}
    for item in processed_items:
        level = item['futurex_level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    logger.info(f"  Questions by level: {dict(sorted(level_counts.items()))}")
    
    logger.info(f"Saved dataset to: {output_file}")


if __name__ == "__main__":
    main()
