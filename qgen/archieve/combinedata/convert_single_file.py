#!/usr/bin/env python3
"""
CLI script to convert JSONL files from question generation format to standardized format.

This script uses the conversion_utils module to:
- Extract question components from final_question field
- Automatically detect news source from filename
- Output converted file with _converted suffix

Usage:
    python convert_single_file.py --input_file /path/to/file.jsonl [--output_file output.jsonl]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgen.utils.io_utils import load_articles_from_file, save_jsonl
from qgen.utils.filtering_utils import is_valid_news_entry
from qgen.utils.conversion_utils import (
    extract_news_source_from_filename,
    convert_question_format,
    standardize_entry_format
)
from qgen.utils.id_utils import remove_fields_from_entries

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL file from question generation format to standardized format"
    )
    
    parser.add_argument(
        '--input_file',
        required=True,
        help='Input JSONL file path'
    )
    
    parser.add_argument(
        '--output_file',
        default=None,
        help='Output JSONL file path (optional, defaults to input_file with _converted suffix)'
    )
    
    parser.add_argument(
        '--news_source',
        default=None,
        help='News source name (optional, auto-detected from filename if not provided)'
    )
    
    parser.add_argument(
        '--skip_invalid',
        action='store_true',
        help='Skip entries that fail validation checks'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / f"{input_path.stem}_converted.jsonl"
    
    # Extract news source from filename if not provided
    if args.news_source:
        news_source = args.news_source
        logger.info(f"Using provided news source: {news_source}")
    else:
        news_source = extract_news_source_from_filename(input_path)
        logger.info(f"Auto-detected news source from filename: {news_source}")
    
    # Load entries
    logger.info(f"Loading entries from {args.input_file}...")
    entries = load_articles_from_file(args.input_file)
    
    if not entries:
        logger.error("No entries found in input file")
        sys.exit(1)
    
    logger.info(f"Loaded {len(entries)} entries")
    
    # Process entries
    converted_entries = []
    skipped_count = 0
    
    for entry in entries:
        # Validate entry if skip_invalid is set
        if args.skip_invalid and not is_valid_news_entry(entry):
            skipped_count += 1
            continue
        
        # Convert and standardize
        try:
            converted = standardize_entry_format(
                entry,
                news_source=news_source,
                original_filename=input_path.name
            )
            converted_entries.append(converted)
        except Exception as e:
            logger.warning(f"Error converting entry: {e}")
            if not args.skip_invalid:
                # If not skipping invalid, keep original entry
                converted_entries.append(entry)
            else:
                skipped_count += 1
    
    logger.info(f"Converted {len(converted_entries)} entries")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} invalid entries")
    
    # Remove unwanted fields
    converted_entries = remove_fields_from_entries(
        converted_entries,
        fields=['resolution_date_response', 'final_question']
    )
    
    # Save output
    logger.info(f"Saving converted entries to {output_path}...")
    save_jsonl(converted_entries, str(output_path))
    
    logger.info(f"Done! Converted {len(converted_entries)} entries")
    logger.info(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
