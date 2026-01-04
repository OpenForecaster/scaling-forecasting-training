#!/usr/bin/env python3
"""
CLI script to clean and filter articles based on various criteria.

This script uses the filtering_utils module to filter entries by:
- Resolution date (optional, if --cutoff_date is provided)
- Answer type validation (strict if --explicit_filter is set)

Usage:
    # Filter by answer type only (no date filtering)
    python clean_articles.py --input_path data/
    
    # Filter by both date and answer type
    python clean_articles.py --input_path data/ --cutoff_date 2025-05-01
    
    # Apply explicit answer type filtering
    python clean_articles.py --input_path data/ --explicit_filter
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgen.utils.io_utils import load_articles_from_file, save_jsonl, get_files_to_process
from qgen.utils.filtering_utils import filter_entries, is_valid_date_format
from qgen.utils.id_utils import remove_fields_from_entries

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_file(
    file_path: str,
    output_dir: Path,
    cutoff_date: str = None,
    explicit_filter: bool = False
):
    """
    Process a single file and save cleaned version.
    
    Args:
        file_path: Path to input file
        output_dir: Directory to save cleaned file
        cutoff_date: Optional cutoff date for filtering
        explicit_filter: Whether to apply explicit answer type filtering
    """
    logger.info(f"Processing file: {file_path}")
    
    # Load articles
    articles = load_articles_from_file(file_path)
    if not articles:
        logger.warning(f"No articles loaded from {file_path}")
        return None
    
    original_count = len(articles)
    logger.info(f"Loaded {original_count} articles")
    
    # Filter entries
    result = filter_entries(
        articles,
        cutoff_date=cutoff_date,
        explicit_answer_filter=explicit_filter
    )
    
    filtered_articles = result['filtered_entries']
    stats = result['stats']
    
    # Remove unwanted fields
    filtered_articles = remove_fields_from_entries(
        filtered_articles,
        fields=['resolution_date_response']
    )
    
    # Create output file path
    input_file = Path(file_path)
    output_file = output_dir / f"{input_file.stem}_cleaned.jsonl"
    
    # Save filtered articles
    save_jsonl(filtered_articles, str(output_file))
    logger.info(f"Saved {len(filtered_articles)} cleaned articles to {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean and filter articles based on resolution date and answer type"
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input directory (processes all .jsonl files) or single .jsonl file"
    )
    
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Optional cutoff date in YYYY-MM-DD format. If provided, filters articles with resolution_date after this date"
    )
    
    parser.add_argument(
        "--explicit_filter",
        action="store_true",
        help="If set, applies explicit answer type filtering with strict keyword matching"
    )
    
    args = parser.parse_args()
    
    # Validate cutoff_date if provided
    if args.cutoff_date:
        if not is_valid_date_format(args.cutoff_date):
            logger.error(f"Invalid cutoff_date format: {args.cutoff_date}. Must be YYYY-MM-DD")
            return
        logger.info(f"Resolution date filtering enabled with cutoff: {args.cutoff_date}")
    else:
        logger.info("Resolution date filtering disabled (no --cutoff_date provided)")
    
    if args.explicit_filter:
        logger.info("Explicit answer type filtering enabled")
    else:
        logger.info("Explicit answer type filtering disabled (accepting all answer types)")
    
    # Get files to process
    files_to_process = get_files_to_process(args.input_path)
    
    if not files_to_process:
        logger.error("No .jsonl files found to process")
        return
    
    # Create cleaned output directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        base_dir = input_path.parent
    else:
        base_dir = input_path
    
    cleaned_dir = base_dir / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    
    logger.info(f"Found {len(files_to_process)} .jsonl files to process")
    logger.info(f"Output directory: {cleaned_dir}")
    
    # Process each file
    total_stats = {
        'total': 0,
        'filtered': 0,
        'valid_date': 0,
        'valid_date_cutoff': 0,
        'valid_answer_type': 0
    }
    
    for file_path in files_to_process:
        try:
            stats = process_file(file_path, cleaned_dir, args.cutoff_date, args.explicit_filter)
            
            if stats:
                # Update total stats
                for key in total_stats:
                    total_stats[key] += stats[key]
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    # Log comprehensive statistics
    logger.info("\n" + "="*60)
    logger.info("CLEANING STATISTICS")
    logger.info("="*60)
    logger.info(f"Total articles processed: {total_stats['total']}")
    
    # Only show date statistics if cutoff_date was provided
    if args.cutoff_date:
        logger.info(f"Articles with valid resolution dates (YYYY-MM-DD): {total_stats['valid_date']}")
        logger.info(f"Articles with resolution date after {args.cutoff_date}: {total_stats['valid_date_cutoff']}")
    
    logger.info(f"Articles with valid answer types: {total_stats['valid_answer_type']}")
    logger.info(f"Final filtered articles: {total_stats['filtered']}")
    
    if total_stats['total'] > 0:
        logger.info(f"Overall filtering rate: {total_stats['filtered']/total_stats['total']*100:.1f}%")
    
    logger.info("="*60)
    logger.info("All files processed and cleaned!")


if __name__ == "__main__":
    main()
