#!/usr/bin/env python3
"""
Script to update resolution_date in a JSONL file to be the minimum of:
- resolution_date
- article_publish_date
- article_modify_date
- article_download_date

Only considers non-None values for the minimum calculation.
"""

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path


def parse_date(date_str):
    """Parse a date string and return a datetime object, or None if parsing fails."""
    if not date_str or date_str is None:
        return None
    
    date_str = str(date_str).strip()
    
    # Handle different date formats
    date_formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%d %H:%M:%S+00:00',
        '%Y-%m-%dT%H:%M:%SZ',  # ISO 8601 with Z suffix
        '%Y-%m-%dT%H:%M:%S',   # ISO 8601 without timezone
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Convert to naive datetime (remove timezone info for comparison)
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    
    # Try to handle various timezone formats using fromisoformat
    try:
        # Handle Z suffix (Zulu time)
        if date_str.endswith('Z'):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        elif '+00:00' in date_str:
            dt = datetime.fromisoformat(date_str.replace('+00:00', '+0000'))
        else:
            dt = datetime.fromisoformat(date_str)
        # Convert to naive datetime (remove timezone info for comparison)
        return dt.replace(tzinfo=None)
    except ValueError:
        print(f"Warning: Could not parse date: {date_str}", file=sys.stderr)
        return None


def update_resolution_date(entry):
    """Update the resolution_date field to be the minimum of available date fields."""
    date_fields = [
        'resolution_date',
        'article_publish_date', 
        'article_modify_date',
        'article_download_date'
    ]
    
    # Parse all available dates
    parsed_dates = []
    for field in date_fields:
        if field in entry and entry[field] is not None:
            parsed_date = parse_date(entry[field])
            if parsed_date is not None:
                parsed_dates.append(parsed_date)
    
    if not parsed_dates:
        print(f"Warning: No valid dates found for entry: {entry.get('question_title', 'Unknown')[:50]}...", file=sys.stderr)
        return entry
    
    # Find the minimum date
    min_date = min(parsed_dates)
    
    # Update the resolution_date field with just the date part (YYYY-MM-DD format)
    entry['resolution_date'] = min_date.strftime('%Y-%m-%d')
    
    return entry


def process_jsonl_file(input_file, output_file):
    """Process the JSONL file and update resolution dates."""
    
    if not Path(input_file).exists():
        print(f"Error: Input file does not exist: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    entries_processed = 0
    entries_updated = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    # Parse JSON line
                    entry = json.loads(line.strip())
                    
                    # Store original resolution_date for comparison
                    original_resolution_date = entry.get('resolution_date')
                    
                    # Update the entry
                    updated_entry = update_resolution_date(entry)
                    
                    # Check if resolution_date was actually changed
                    if updated_entry.get('resolution_date') != original_resolution_date:
                        entries_updated += 1
                    
                    # Write updated entry to output file
                    json.dump(updated_entry, outfile, ensure_ascii=False)
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
    print(f"Entries with updated resolution_date: {entries_updated}", file=sys.stderr)
    print(f"Output written to: {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Update resolution_date in JSONL file to minimum of available date fields"
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default='/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_fivenewscombined_1000.jsonl',
        help='Input JSONL file path (default: /fast/nchandak/forecasting/newsdata/testset/o4-mini-high_fivenewscombined_1000.jsonl)'
    )
    args = parser.parse_args()
    # make output file name the same as input file but with _updated_resolution.jsonl
    output_file = args.input_file.replace('.jsonl', '_updated_resolution.jsonl')
    process_jsonl_file(args.input_file, output_file)


if __name__ == '__main__':
    main()
