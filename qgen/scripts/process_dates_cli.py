#!/usr/bin/env python3
"""
Process Dates CLI - Unified interface for date operations on questions.

This script consolidates all date-related operations:
- Extract resolution dates using OpenRouter (default) or local model
- Extract and remove start dates from backgrounds
- Update resolution dates to minimum of available dates
- Update start dates to minimum of available dates

Usage:
    # Extract resolution dates (uses OpenRouter by default)
    python scripts/process_dates_cli.py \\
        --input_path questions.jsonl \\
        --extract_resolution
    
    # Extract resolution dates with local model
    python scripts/process_dates_cli.py \\
        --input_path questions.jsonl \\
        --local_model_path /path/to/model \\
        --extract_resolution
    
    # Update all dates
    python scripts/process_dates_cli.py \\
        --input_path questions.jsonl \\
        --extract_start --update_resolution --update_start
"""

import os
import argparse
import logging
import sys

# Add parent of qgen directory to path so we can import qgen package
qgen_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(qgen_dir)
sys.path.insert(0, parent_dir)

from qgen.processors.date_processor import DateProcessor
from qgen.inference.openrouter_inference import OpenRouterInference

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process dates in forecasting questions")
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input file or directory"
    )
    
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Path to local VLLM model (if not specified, uses OpenRouter with llama-4-maverick)"
    )
    
    parser.add_argument(
        "--openrouter_model",
        type=str,
        default="meta-llama/llama-4-maverick",
        help="OpenRouter model to use for date extraction (default: meta-llama/llama-4-maverick)"
    )
    
    parser.add_argument(
        "--extract_resolution",
        action="store_true",
        help="Extract resolution dates using VLLM"
    )
    
    parser.add_argument(
        "--extract_start",
        action="store_true",
        help="Extract and remove start dates from backgrounds"
    )
    
    parser.add_argument(
        "--update_resolution",
        action="store_true",
        help="Update resolution dates to minimum of available dates"
    )
    
    parser.add_argument(
        "--update_start",
        action="store_true",
        help="Update start dates to minimum of available dates"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.extract_resolution, args.extract_start, args.update_resolution, args.update_start]):
        logger.error("At least one operation flag must be specified")
        return
    
    # Initialize inference engine
    inference_engine = None
    model_path = None
    
    if args.extract_resolution:
        if args.local_model_path:
            # Use local VLLM model
            logger.info(f"Using local VLLM model: {args.local_model_path}")
            model_path = args.local_model_path
        else:
            # Use OpenRouter by default
            logger.info(f"Using OpenRouter with model: {args.openrouter_model}")
            inference_engine = OpenRouterInference(
                model=args.openrouter_model,
                max_tokens=4096,
                temperature=0.3
            )
    
    # Initialize processor
    processor = DateProcessor(model_path=model_path, inference_engine=inference_engine)
    
    # Process files
    processor.process_directory(
        input_path=args.input_path,
        extract_resolution=args.extract_resolution,
        extract_start=args.extract_start,
        update_resolution=args.update_resolution,
        update_start=args.update_start
    )
    
    logger.info("Date processing complete!")


if __name__ == "__main__":
    main()


