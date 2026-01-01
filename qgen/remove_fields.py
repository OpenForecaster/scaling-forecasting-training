#!/usr/bin/env python3
"""
Script to remove specific fields from a JSONL file.
"""

import json
import argparse
import sys
from pathlib import Path


def remove_fields_from_jsonl(input_file, output_file, fields_to_remove):
    """Remove specified fields from each entry in a JSONL file."""
    
    if not Path(input_file).exists():
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    entries_processed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    # Parse JSON line
                    entry = json.loads(line.strip())
                    
                    # Remove specified fields
                    for field in fields_to_remove:
                        if field in entry:
                            del entry[field]
                    
                    # Write updated entry to output file
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                    entries_processed += 1
                    
                    if entries_processed % 100 == 0:
                        print(f"Processed {entries_processed} entries...", file=sys.stderr)
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                    continue
    
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing complete!", file=sys.stderr)
    print(f"Total entries processed: {entries_processed}", file=sys.stderr)
    print(f"Output written to: {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Remove specified fields from JSONL file"
    )
    
    parser.add_argument(
        'input_file',
        help='Input JSONL file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: input_file_no_fields.jsonl)'
    )
    
    parser.add_argument(
        '-f', '--fields',
        nargs='+',
        default=['reasoning'],
        help='Fields to remove (default: response reasoning)'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_file)
        args.output = input_path.parent / f"{input_path.stem}_no_fields{input_path.suffix}"
    
    print(f"Removing fields: {args.fields}", file=sys.stderr)
    remove_fields_from_jsonl(args.input_file, args.output, args.fields)


if __name__ == '__main__':
    main() 