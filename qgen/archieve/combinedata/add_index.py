#!/usr/bin/env python3
"""
CLI script to add random unique question_id field to JSONL files.

This script uses the id_utils module to add random unique IDs to entries.

Usage:
    python add_index.py --input_file input.jsonl [--output_file output.jsonl]
    
If output file is not specified, it will modify the input file in-place.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgen.utils.io_utils import load_articles_from_file, save_jsonl
from qgen.utils.id_utils import add_ids_to_entries


def main():
    parser = argparse.ArgumentParser(
        description="Add random unique question_id field to JSONL files where it's missing"
    )
    parser.add_argument(
        '--input_file',
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output_file', 
        default=None,
        help='Output JSONL file path (optional, defaults to modifying input file)'
    )
    parser.add_argument(
        '--start-idx', 
        type=int, 
        default=0,
        help='Minimum value for random IDs (default: 0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible ID generation (optional)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Load entries
    print(f"Loading entries from {args.input_file}...")
    entries = load_articles_from_file(args.input_file)
    
    if not entries:
        print("No entries found in input file.")
        sys.exit(1)
    
    print(f"Loaded {len(entries)} entries")
    
    # Dry run mode
    if args.dry_run:
        print("\nDRY RUN MODE - No files will be modified")
        
        existing_count = sum(1 for entry in entries if 'question_id' in entry)
        needing_count = len(entries) - existing_count
        
        print(f"Entries already with question_id: {existing_count}")
        print(f"Entries needing question_id: {needing_count}")
        
        if existing_count > 0:
            existing_ids = [e['question_id'] for e in entries if 'question_id' in e]
            print(f"Existing IDs range: {min(existing_ids)} to {max(existing_ids)}")
        
        return
    
    # Add IDs to entries
    print(f"\nAdding question_id to entries...")
    if args.seed is not None:
        print(f"Using random seed: {args.seed}")
    
    # Count before
    entries_with_id_before = sum(1 for entry in entries if 'question_id' in entry)
    
    # Add IDs
    entries = add_ids_to_entries(
        entries,
        id_field='question_id',
        start_idx=args.start_idx,
        seed=args.seed,
        remove_fields=['resolution_date_response']  # Clean up unwanted fields
    )
    
    # Count after
    entries_with_id_after = sum(1 for entry in entries if 'question_id' in entry)
    entries_added = entries_with_id_after - entries_with_id_before
    
    print(f"Added question_id to {entries_added} entries")
    
    # Save output
    output_path = args.output_file if args.output_file else args.input_file
    print(f"\nSaving to {output_path}...")
    save_jsonl(entries, output_path)
    
    print(f"Done! Processed {len(entries)} entries.")
    if args.output_file:
        print(f"Output written to: {args.output_file}")
    else:
        print(f"Input file modified in-place: {args.input_file}")


if __name__ == "__main__":
    main()
