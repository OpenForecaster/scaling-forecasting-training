#!/usr/bin/env python3
"""
End-to-end pipeline for generating, converting, filtering, and indexing forecasting questions.

This script orchestrates the complete workflow:
1. Generate questions from articles (free-form, with leakage checking, best selection, validation)
2. Convert to standardized format (conversion_utils)
3. Extract and update resolution/start dates (dates updated to minimum of available dates)
4. Filter by date and answer type (filtering_utils)
5. Add unique question IDs (id_utils)

All intermediate outputs are saved for inspection.

Quality defaults (always enabled):
- Free-form questions 
- Leakage checking and removal
- Best question selection from multiple candidates
- Question validation
- Date updates to minimum of available dates

Usage:
    python run_pipeline.py \\
        --article_path articles.jsonl \\
        --output_dir /path/to/output \\
        --use_openrouter --creator_model deepseek/deepseek-chat-v3-0324 --selector_model meta-llama/llama-4-maverick \\
        --first_date 2025-01-01 \\
        --seed 42
"""

import os
import sys
import argparse
import asyncio
import logging
import subprocess
from pathlib import Path

# Add parent directory to path so we can import qgen
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from qgen.utils import (
    load_articles_from_file,
    save_jsonl,
    extract_news_source_from_filename,
    standardize_entry_format,
    filter_entries,
    add_ids_to_entries,
    remove_fields_from_entries,
    is_valid_news_entry
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_question_generation(args, output_dir):
    """
    Step 1: Generate questions using generate_questions.py
    
    Always enables: --freeq, --check_leakage, --choose_best, --validate
    
    Returns:
        Path to generated questions file
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: GENERATING QUESTIONS")
    logger.info("="*70)
    
    # Determine output path for generated questions
    article_basename = Path(args.article_path).stem
    generated_file = output_dir / f"{article_basename}_generated.jsonl"
    
    # Build command (use absolute path for robustness)
    generate_script = os.path.join(script_dir, "scripts", "generate_questions.py")
    cmd = [
        sys.executable,
        generate_script,
        "--article_path", args.article_path,
        "--output_path", str(generated_file),
        "--num_q_per_article", str(args.num_q_per_article),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--batch_size", str(args.batch_size)
    ]
    
    # Add default flags (always enabled in pipeline)
    cmd.extend(["--freeq", "--check_leakage", "--choose_best", "--validate"])
    
    # Add optional flags
    if args.regenerate:
        cmd.append("--regenerate")
    
    # Add model configuration
    if args.use_openrouter:
        cmd.extend([
            "--use_openrouter", 
            "--creator_model", args.creator_model,
            "--selector_model", args.selector_model
        ])
    else:
        if not args.model_path:
            raise ValueError("--model_path required when not using OpenRouter")
        cmd.extend(["--model_path", args.model_path])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    logger.info(f"✓ Generated questions saved to: {generated_file}")
    return generated_file


def convert_to_standard_format(generated_file, output_dir):
    """
    Step 2: Convert generated questions to standardized format
    
    Returns:
        Path to converted file
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: CONVERTING TO STANDARD FORMAT")
    logger.info("="*70)
    
    # Load generated questions
    entries = load_articles_from_file(str(generated_file))
    logger.info(f"Loaded {len(entries)} generated questions")
    
    # Extract news source from original filename
    news_source = extract_news_source_from_filename(generated_file)
    logger.info(f"Detected news source: {news_source}")
    
    # Convert and standardize entries
    converted_entries = []
    skipped = 0
    
    for entry in entries:
        try:
            # Check if entry is already in standard format (has question_title, background, etc.)
            if 'question_title' in entry and 'background' in entry:
                # Already in standard format, just add metadata if missing
                if 'news_source' not in entry:
                    entry['news_source'] = news_source
                if 'data_source' not in entry:
                    entry['data_source'] = 'news_generated'
                if 'original_file' not in entry:
                    entry['original_file'] = generated_file.name
                
                # Ensure article fields are preserved (copy from non-prefixed to prefixed if needed)
                if 'article_title' not in entry and 'title' in entry:
                    entry['article_title'] = entry['title']
                if 'article_description' not in entry and 'description' in entry:
                    entry['article_description'] = entry['description']
                if 'article_maintext' not in entry and 'maintext' in entry:
                    entry['article_maintext'] = entry['maintext']
                if 'article_publish_date' not in entry and 'date_publish' in entry:
                    entry['article_publish_date'] = entry['date_publish']
                if 'article_modify_date' not in entry and 'date_modify' in entry:
                    entry['article_modify_date'] = entry['date_modify']
                if 'article_download_date' not in entry and 'date_download' in entry:
                    entry['article_download_date'] = entry['date_download']
                
                converted_entries.append(entry)
            elif 'final_question' in entry:
                # Needs conversion from XML format
                if not is_valid_news_entry(entry):
                    skipped += 1
                    continue
                converted = standardize_entry_format(
                    entry,
                    news_source=news_source,
                    original_filename=generated_file.name
                )
                converted_entries.append(converted)
            else:
                # Invalid entry
                skipped += 1
                continue
        except Exception as e:
            logger.warning(f"Error converting entry: {e}")
            skipped += 1
    
    logger.info(f"Converted {len(converted_entries)} entries (skipped {skipped} invalid)")
    
    # Save converted file
    converted_file = output_dir / f"{generated_file.stem}_converted.jsonl"
    save_jsonl(converted_entries, str(converted_file))
    logger.info(f"✓ Converted questions saved to: {converted_file}")
    
    return converted_file


def extract_dates(converted_file, output_dir, args):
    """
    Step 3: Extract resolution and start dates
    
    Always enables: --update_resolution, --update_start (dates updated to minimum)
    
    Returns:
        Path to file with extracted dates
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 3: EXTRACTING DATES")
    logger.info("="*70)
    
    # Build command for date extraction (use absolute path)
    date_script = os.path.join(script_dir, "scripts", "process_dates_cli.py")
    cmd = [
        sys.executable,
        date_script,
        "--input_path", str(converted_file),
        "--extract_resolution"
    ]
    
    # Add optional arguments for date extraction
    if args.extract_start_dates:
        cmd.append("--extract_start")
    
    # Always update resolution and start dates to minimum of available dates
    cmd.extend(["--update_resolution", "--update_start"])
    
    # Add model configuration (uses OpenRouter by default)
    if args.local_model_path:
        cmd.extend(["--local_model_path", args.local_model_path])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    # The date extraction script creates a file with _date_extracted.jsonl suffix
    dated_file = output_dir / f"{converted_file.stem}_date_extracted.jsonl"
    
    if not dated_file.exists():
        raise FileNotFoundError(f"Expected output file not found: {dated_file}")
    
    logger.info(f"✓ Dates extracted and saved to: {dated_file}")
    return dated_file


def filter_questions(dated_file, output_dir, args):
    """
    Step 4: Filter questions by date and answer type
    
    Also filters out questions without valid resolution dates.
    
    Returns:
        Path to filtered file
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 4: FILTERING QUESTIONS")
    logger.info("="*70)
    
    # Load questions
    entries = load_articles_from_file(str(dated_file))
    logger.info(f"Loaded {len(entries)} questions with dates")
    
    # First filter: Remove entries without valid resolution dates
    from qgen.utils.filtering_utils import is_valid_date_format
    entries_with_valid_dates = []
    invalid_date_count = 0
    
    for entry in entries:
        resolution_date = entry.get('resolution_date', '')
        if resolution_date and is_valid_date_format(resolution_date):
            entries_with_valid_dates.append(entry)
        else:
            invalid_date_count += 1
    
    logger.info(f"Filtered out {invalid_date_count} entries with invalid/missing resolution dates")
    logger.info(f"Remaining entries with valid dates: {len(entries_with_valid_dates)}")
    
    # Second filter: Apply date cutoff and answer type filters
    result = filter_entries(
        entries_with_valid_dates,
        first_date=args.first_date if hasattr(args, 'first_date') else None,
        explicit_answer_filter=args.explicit_filter if hasattr(args, 'explicit_filter') else False
    )
    
    filtered_entries = result['filtered_entries']
    stats = result['stats']
    
    # Log statistics
    logger.info(f"Total entries: {stats['total']}")
    if args.first_date:
        logger.info(f"Valid dates (YYYY-MM-DD): {stats['valid_date']}")
        logger.info(f"On or after first date ({args.first_date}): {stats['valid_date_cutoff']}")
    logger.info(f"Valid answer types: {stats['valid_answer_type']}")
    logger.info(f"Final filtered: {stats['filtered']}")
    
    # Remove unwanted fields
    filtered_entries = remove_fields_from_entries(
        filtered_entries,
        fields=['resolution_date_response', 'final_question']
    )
    
    # Save filtered file
    filtered_file = output_dir / f"{dated_file.stem}_filtered.jsonl"
    save_jsonl(filtered_entries, str(filtered_file))
    logger.info(f"✓ Filtered questions saved to: {filtered_file}")
    
    return filtered_file


def add_question_ids(filtered_file, output_dir, args):
    """
    Step 5: Add unique question IDs
    
    Returns:
        Path to final file with IDs
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 5: ADDING QUESTION IDs")
    logger.info("="*70)
    
    # Load filtered questions
    entries = load_articles_from_file(str(filtered_file))
    logger.info(f"Loaded {len(entries)} filtered questions")
    
    # Add IDs
    entries = add_ids_to_entries(
        entries,
        id_field='question_id',
        seed=args.seed if hasattr(args, 'seed') else None
    )
    
    logger.info(f"Added question_id to all {len(entries)} entries")
    
    # Save final file
    final_file = output_dir / f"{Path(args.article_path).stem}_final_questions.jsonl"
    save_jsonl(entries, str(final_file))
    logger.info(f"✓ Final questions saved to: {final_file}")
    
    return final_file


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for question generation and processing.\n"
                    "Quality defaults always enabled: freeq, leakage checking, best selection, "
                    "validation, and date updates.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--article_path",
        required=True,
        help="Path to news article file or directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save all output files (REQUIRED)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to local HuggingFace model for vLLM inference"
    )
    parser.add_argument(
        "--use_openrouter",
        action="store_true",
        help="Use OpenRouter for inference"
    )
    parser.add_argument(
        "--creator_model",
        default="deepseek/deepseek-chat-v3-0324",
        help="Model to use for question generation (creator) (default: deepseek/deepseek-chat-v3-0324)"
    )
    parser.add_argument(
        "--selector_model",
        default="meta-llama/llama-4-maverick",
        help="Model to use for selection/validation (selector) (default: meta-llama/llama-4-maverick)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_q_per_article",
        type=int,
        default=1,
        help="Number of questions to generate per article (default: 1)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum tokens for generation (default: 16384)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for parallel processing (default: 1000)"
    )
    
    # Processing flags (most are enabled by default)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore existing results and regenerate all questions"
    )
    
    # Date extraction parameters
    parser.add_argument(
        "--extract_start_dates",
        action="store_true",
        help="Also extract and remove start dates from backgrounds"
    )
    parser.add_argument(
        "--local_model_path",
        default=None,
        help="Local model path for date extraction (if not using OpenRouter)"
    )
    
    # Filtering parameters
    parser.add_argument(
        "--first_date",
        default=None,
        help="First date for filtering (YYYY-MM-DD). If provided, only keeps questions with resolution_date on or after this"
    )
    parser.add_argument(
        "--explicit_filter",
        action="store_true",
        help="Apply explicit answer type filtering with strict keyword matching"
    )
    
    # ID generation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible ID generation"
    )
    
    args = parser.parse_args()
    
    # Validate output_dir
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("FORECASTING QUESTIONS GENERATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Input: {args.article_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Quality defaults: freeq, leakage check, best selection, validation, date updates")
    logger.info("="*70)
    
    try:
        # Step 1: Generate questions
        generated_file = run_question_generation(args, output_dir)
        
        # Step 2: Convert to standard format
        converted_file = convert_to_standard_format(generated_file, output_dir)
        
        # Step 3: Extract dates
        dated_file = extract_dates(converted_file, output_dir, args)
        
        # Step 4: Filter questions
        filtered_file = filter_questions(dated_file, output_dir, args)
        
        # Step 5: Add question IDs
        final_file = add_question_ids(filtered_file, output_dir, args)
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info("Intermediate files:")
        logger.info(f"  1. Generated:  {generated_file.name}")
        logger.info(f"  2. Converted:  {converted_file.name}")
        logger.info(f"  3. Dated:      {dated_file.name}")
        logger.info(f"  4. Filtered:   {filtered_file.name}")
        logger.info("="*70)
        logger.info(f"FINAL OUTPUT: {final_file}")
        logger.info("="*70)
        
        # Print just the filename for easy copying
        print(f"\n>>> Final questions file: {final_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

