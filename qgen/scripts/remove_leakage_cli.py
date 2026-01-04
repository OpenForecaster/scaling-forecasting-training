#!/usr/bin/env python3
"""
Remove Leakage CLI - Unified interface for leakage detection and removal.

This script provides three approaches to handling leakage:
1. LLM-based leakage removal
2. Pattern-based exact leakage removal
3. Filtering questions with detected leakage

Usage:
    # LLM-based removal
    python scripts/remove_leakage_cli.py \\
        --mode llm \\
        --input_file questions.jsonl \\
        --batch_size 5
    
    # Pattern-based removal
    python scripts/remove_leakage_cli.py \\
        --mode pattern \\
        --input_path questions.jsonl
    
    # Filter out leaky questions
    python scripts/remove_leakage_cli.py \\
        --mode filter \\
        --input_file questions.jsonl \\
        --output_file questions_filtered.jsonl
"""

import os
import argparse
import asyncio
import logging
import sys

# Add parent of qgen directory to path so we can import qgen package
qgen_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(qgen_dir)
sys.path.insert(0, parent_dir)

from qgen.filters.leakage_filter import (
    LeakageRemover, 
    filter_by_leakage, 
    remove_exact_leakage_patterns,
    remove_exact_leakage_from_entries
)
from qgen.inference.openrouter_inference import OpenRouterInference
from qgen.utils.io_utils import load_articles_from_file, save_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Remove leakage from forecasting questions")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["llm", "pattern", "filter"],
        required=True,
        help="Mode: 'llm' for LLM-based removal, 'pattern' for regex patterns, 'filter' to filter out leaky questions"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input JSONL file"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input directory (processes all .jsonl files)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output file (for filter mode)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for LLM processing"
    )
    
    parser.add_argument(
        "--use_freeq",
        type=bool,
        default=True,
        help="Process free-form questions"
    )
    
    args = parser.parse_args()
    
    if args.mode == "llm":
        # LLM-based leakage removal
        if not args.input_file:
            logger.error("--input_file required for llm mode")
            return
        
        # Initialize inference engine
        inference_engine = OpenRouterInference(
            model="meta-llama/llama-4-maverick",
            max_tokens=10000,
            temperature=0.6
        )
        
        # Initialize remover
        remover = LeakageRemover(
            inference_engine=inference_engine,
            use_freeq=args.use_freeq
        )
        
        # Load and process
        entries = remover.load_jsonl(args.input_file)
        if entries:
            await remover.remove_leakage_from_entries(
                entries, 
                input_file=args.input_file,
                batch_size=args.batch_size
            )
            remover.save_jsonl(entries, args.input_file)
        
    elif args.mode == "pattern":
        # Pattern-based exact leakage removal
        if not args.input_path:
            logger.error("--input_path required for pattern mode")
            return
        
        from qgen.utils.io_utils import get_files_to_process
        
        files = get_files_to_process(args.input_path)
        for file_path in files:
            logger.info(f"\nProcessing {file_path}...")
            articles = load_articles_from_file(file_path)
            
            # Use the wrapper function which includes statistics
            cleaned_articles = remove_exact_leakage_from_entries(articles)
            
            save_jsonl(cleaned_articles, file_path)
            logger.info(f"Saved cleaned data to {file_path}")
    
    elif args.mode == "filter":
        # Filter out questions with leakage
        if not args.input_file or not args.output_file:
            logger.error("--input_file and --output_file required for filter mode")
            return
        
        logger.info(f"Loading entries from {args.input_file}...")
        entries = load_articles_from_file(args.input_file)
        
        # filter_by_leakage now logs statistics internally
        filtered, removed_count = filter_by_leakage(entries)
        
        save_jsonl(filtered, args.output_file)
        logger.info(f"\nSaved {len(filtered)} clean entries to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())


