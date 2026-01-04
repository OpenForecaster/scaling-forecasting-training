#!/usr/bin/env python3
"""
Prepare custom question files for VERL training.

This script:
1. Loads questions from a JSONL file
2. Filters out irrelevant/invalid questions
3. Converts to VERL format (parquet)
4. Saves to specified output directory

Usage:
    python prepare_custom_dataset.py --questions_file /path/to/questions.jsonl --output_dir data/
    python prepare_custom_dataset.py --questions_file questions.jsonl --subsample 500 --output_dir data/
"""

import argparse
import logging
import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path

import datasets

from prompt_utils import format_forecasting_prompt, format_forecasting_prompt_binary

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_questions_from_jsonl(file_path: str) -> List[Dict]:
    """
    Load questions from a JSONL file and filter out invalid entries.
    
    Args:
        file_path: Path to the JSONL file containing questions
        
    Returns:
        List of valid question dictionaries
    """
    questions_data = []
    
    logger.info(f"Loading questions from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    
                    # Skip if question is not relevant or the article is not relevant
                    if 'question_relevant' in entry and int(entry['question_relevant']) == 0:
                        continue
                    
                    if 'article_relevant' in entry and int(entry['article_relevant']) == 0:
                        continue
                    
                    if 'no_good_question' in entry and int(entry['no_good_question']) == 1:
                        continue
                    
                    # Extract question title
                    question_title = entry.get('question_title', entry.get('question', ''))
                    
                    # Only add if we have a valid question title
                    if question_title and question_title.strip():
                        questions_data.append(entry)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data


def process_question(entry: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """
    Process a single question into VERL format.
    
    Args:
        entry: Dictionary containing question data
        split: Dataset split ("train", "validation", or "test")
        idx: Index of the question in the dataset
        
    Returns:
        Dictionary in the format expected by VERL training pipeline
    """
    # Extract question components
    question_title = entry.get('question_title', entry.get('question', ''))
    background = entry.get('background', '')
    resolution_criteria = entry.get('resolution_criteria', '')
    if not resolution_criteria or len(resolution_criteria.strip()) == 0:
        resolution_criteria = "N/A"
    
    answer = entry.get('answer', '')
    answer_type = entry.get('answer_type', '')
    resolution_date = entry.get('resolution_date', '')
    question_start_date = entry.get('question_start_date', entry.get('date_begin', ''))
    question_id = entry.get('question_id', entry.get('question_idx', ''))
    data_source = entry.get('data_source', 'custom')
    news_source = entry.get('news_source', 'unknown')
    
    # Extract resolution from answer if not explicitly provided
    resolution = entry.get('resolution', -1)
    if resolution == -1:
        if "yes" in answer.lower():
            resolution = 1
        elif "no" in answer.lower():
            resolution = 0
    
    # Determine if this is a binary question
    is_binary = 'binary' in str(answer_type).lower() or resolution in [0, 1]
    
    # Determine data field based on source
    if "metaculus" in str(data_source).lower():
        data_field = f"binary/metaculus-{split}"
    elif "manifold" in str(data_source).lower():
        data_field = f"binary/manifold-{split}"
    elif "theguardian" in str(news_source).lower():
        data_field = f"freeform/theguardian-{split}"
    elif is_binary:
        data_field = f"binary/custom-{split}"
    else:
        data_field = f"freeform/custom-{split}"
    
    # Format the prompt
    if is_binary:
        prompt_text = format_forecasting_prompt_binary(
            question_title, background, resolution_criteria, question_start_date
        )
    else:
        prompt_text = format_forecasting_prompt(
            question_title, background, resolution_criteria, answer_type
        )
    
    # Create VERL format
    data = {
        "data_source": data_field,
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "forecasting",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer_type": answer_type,
            "question_idx": str(question_id),
            "answer": answer,
            "question": question_title,
            "background": background,
            "resolution_criteria": resolution_criteria,
            "resolution_date": resolution_date,
            "question_source": data_field,
            "resolution": int(resolution),
        },
    }
    
    return data


def save_to_parquet(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the processed dataset to a parquet file.
    
    Args:
        data: List of dictionaries, each representing one example
        output_path: Path where the parquet file should be saved
    """
    logger.info(f"Converting {len(data)} examples to HuggingFace Dataset...")
    
    # Convert list of dicts to HuggingFace Dataset
    dataset = datasets.Dataset.from_list(data)
    
    logger.info(f"Saving dataset to {output_path}")
    dataset.to_parquet(output_path)
    
    logger.info(f"Successfully saved dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare custom question file for VERL training"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        required=True,
        help="Path to JSONL file containing questions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Directory to save output parquet file (default: data/)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Number of samples to randomly subsample from the dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsampling (default: 42)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split designation (default: train)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load questions from JSONL file
    questions_data = load_questions_from_jsonl(args.questions_file)
    
    if not questions_data:
        logger.error("No valid questions found in input file")
        return
    
    # Subsample if requested
    if args.subsample is not None and args.subsample < len(questions_data):
        logger.info(f"Subsampling {args.subsample} questions from {len(questions_data)}")
        questions_data = random.sample(questions_data, args.subsample)
        logger.info(f"Subsampled to {len(questions_data)} questions")
    
    # Shuffle the data for randomness
    random.shuffle(questions_data)
    logger.info(f"Shuffled {len(questions_data)} questions")
    
    # Process all questions into VERL format
    logger.info("Processing questions into VERL format...")
    processed_data = []
    
    for idx, entry in enumerate(questions_data):
        try:
            processed_entry = process_question(entry, args.split, idx)
            processed_data.append(processed_entry)
        except Exception as e:
            logger.warning(f"Failed to process question {idx}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} questions")
    
    # Generate output filename
    input_filename = Path(args.questions_file).stem
    subsample_suffix = f"_samples{args.subsample}" if args.subsample else ""
    output_filename = f"{input_filename}_verl{subsample_suffix}.parquet"
    output_path = output_dir / output_filename
    
    # Save to parquet
    save_to_parquet(processed_data, str(output_path))
    
    # Calculate statistics
    binary_count = sum(1 for d in processed_data if 'binary' in d['data_source'])
    freeform_count = len(processed_data) - binary_count
    
    # Print summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Input file: {args.questions_file}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Total questions processed: {len(processed_data)}")
    logger.info(f"  - Binary questions: {binary_count}")
    logger.info(f"  - Freeform questions: {freeform_count}")
    logger.info(f"Output file: {output_path}")
    logger.info("="*70)
    
    print(f"\n✓ Successfully created: {output_path}")


if __name__ == "__main__":
    main()

