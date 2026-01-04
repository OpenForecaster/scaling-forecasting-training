#!/usr/bin/env python3
"""
Load OpenForesight dataset from HuggingFace and prepare it for VERL training.

This script:
1. Loads OpenForesight from HuggingFace
2. Uses the 'prompt' field directly if available, otherwise constructs from components
3. Optionally subsamples the dataset
4. Converts to VERL format (parquet)
5. Saves to specified output directory

Usage:
    python load_foresight.py --split train --subsample 1000 --output_dir data/
    python load_foresight.py --split validation --output_dir data/
"""

import argparse
import logging
import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path

import datasets
from datasets import load_dataset
from prompt_utils import format_forecasting_prompt, format_forecasting_prompt_binary

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_foresight_example(example: Dict[str, Any], split: str, idx: int) -> Dict[str, Any]:
    """
    Process a single OpenForesight example into VERL format.
    
    Args:
        example: Dictionary containing question data from OpenForesight
        split: Dataset split ("train", "validation", or "test")
        idx: Index of the question in the dataset
        
    Returns:
        Dictionary in the format expected by VERL training pipeline
    """
    # Extract fields from OpenForesight dataset
    question_title = example.get('question_title', example.get('question', ''))
    background = example.get('background', '')
    resolution_criteria = example.get('resolution_criteria', 'N/A')
    answer = example.get('answer', '')
    answer_type = example.get('answer_type', '')
    resolution_date = example.get('resolution_date', '')
    question_start_date = example.get('question_start_date', '')
    data_source = example.get('data_source', 'openforesight')
    
    # Check if the dataset already has a formatted prompt
    # The 'prompt' field in OpenForesight is the pre-formatted prompt, NOT the article
    existing_prompt = example.get('prompt', '')
    
    # Determine if this is a binary question
    is_binary = 'binary' in str(answer_type).lower() or answer.lower() in ['yes', 'no']
    
    # Use existing prompt if available, otherwise construct it
    if existing_prompt and len(existing_prompt) > 100:
        # Use the pre-formatted prompt from the dataset
        prompt_text = existing_prompt
        logger.debug(f"Using existing prompt for example {idx}")
    else:
        # Construct the prompt from components
        logger.debug(f"Constructing prompt for example {idx}")
        if is_binary:
            prompt_text = format_forecasting_prompt_binary(
                question_title, background, resolution_criteria, question_start_date
            )
        else:
            prompt_text = format_forecasting_prompt(
                question_title, background, resolution_criteria, answer_type
            )
    
    # Determine data field
    if is_binary:
        data_field = f"binary/openforesight-{split}"
    else:
        data_field = f"freeform/openforesight-{split}"
    
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
            "answer": answer,
            "question": question_title,
            "background": background,
            "resolution_criteria": resolution_criteria,
            "resolution_date": resolution_date,
            "question_source": data_field,
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
        description="Load OpenForesight dataset and convert to VERL format"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to load (default: train)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Number of samples to randomly subsample from the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Directory to save output parquet file (default: data/)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load OpenForesight dataset from HuggingFace
    logger.info(f"Loading OpenForesight dataset, split: {args.split}")
    try:
        dataset = load_dataset('nikhilchandak/OpenForesight', split=args.split)
        logger.info(f"Loaded {len(dataset)} examples from OpenForesight")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Subsample if requested
    if args.subsample is not None and args.subsample < len(dataset):
        logger.info(f"Subsampling {args.subsample} examples from {len(dataset)}")
        indices = random.sample(range(len(dataset)), args.subsample)
        indices.sort()  # Keep them in order for reproducibility
        dataset = dataset.select(indices)
        logger.info(f"Subsampled to {len(dataset)} examples")
    
    # Process all examples
    logger.info("Processing examples into VERL format...")
    processed_data = []
    
    for idx, example in enumerate(dataset):
        try:
            processed_example = process_foresight_example(example, args.split, idx)
            processed_data.append(processed_example)
        except Exception as e:
            logger.warning(f"Failed to process example {idx}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} examples")
    
    # Generate output filename
    subsample_suffix = f"_samples{args.subsample}" if args.subsample else ""
    output_filename = f"foresight_{args.split}{subsample_suffix}.parquet"
    output_path = output_dir / output_filename
    
    # Save to parquet
    save_to_parquet(processed_data, str(output_path))
    
    # Print summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Split: {args.split}")
    logger.info(f"Total examples: {len(processed_data)}")
    logger.info(f"Output file: {output_path}")
    logger.info("="*70)
    
    print(f"\n✓ Successfully created: {output_path}")


if __name__ == "__main__":
    main()

