#!/usr/bin/env python3
"""
Script to add question_idx field to JSONL files where it's missing.
Usage: python add_index.py input.jsonl [output.jsonl]
If output file is not specified, it will modify the input file in-place.
"""

import json
import argparse
import sys
from pathlib import Path


def add_question_idx(input_file, output_file=None, start_idx=0):
    """
    Add question_idx field to JSONL lines that don't have it.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str, optional): Path to output JSONL file. If None, modifies input file.
        start_idx (int): Starting index value (default: 0)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    # Read all lines and process them
    lines = []
    current_idx = start_idx
    
    print(f"Processing {input_file}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Remove resolution_date_response if it exists
                if 'resolution_date_response' in data:
                    del data['resolution_date_response']
                    print(f"Line {line_num}: Dropped 'resolution_date_response' field")

                # Check if question_idx already exists
                if 'question_idx' not in data:
                    data['question_idx'] = current_idx
                    print(f"Line {line_num}: Added question_idx = {current_idx}")
                    current_idx += 1
                else:
                    print(f"Line {line_num}: question_idx already exists ({data['question_idx']})")

                lines.append(json.dumps(data, ensure_ascii=False))

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Problematic line: {line}")
                sys.exit(1)

    # Write output
    output_path = Path(output_file) if output_file else input_path
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Done! Processed {len(lines)} lines.")
    if output_file:
        print(f"Output written to: {output_file}")
    else:
        print(f"Input file modified in-place: {input_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add question_idx field to JSONL files where it's missing"
    )
    parser.add_argument(
        '--input_file', 
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output_file', 
        nargs='?', 
        default=None,
        help='Output JSONL file path (optional, defaults to modifying input file)'
    )
    parser.add_argument(
        '--start-idx', 
        type=int, 
        default=0,
        help='Starting index value (default: 0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        input_path = Path(args.input_file)
        
        if not input_path.exists():
            print(f"Error: Input file '{args.input_file}' does not exist.")
            sys.exit(1)
        
        lines_needing_idx = 0
        lines_with_idx = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'question_idx' not in data:
                        lines_needing_idx += 1
                        print(f"Line {line_num}: Would add question_idx")
                    else:
                        lines_with_idx += 1
                        print(f"Line {line_num}: Already has question_idx ({data['question_idx']})")
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
        
        print(f"\nSummary:")
        print(f"Lines needing question_idx: {lines_needing_idx}")
        print(f"Lines already with question_idx: {lines_with_idx}")
        print(f"Total lines: {lines_needing_idx + lines_with_idx}")
        
    else:
        add_question_idx(args.input_file, args.output_file, args.start_idx)


if __name__ == "__main__":
    main()
