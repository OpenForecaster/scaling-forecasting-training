#!/usr/bin/env python3
"""
Script to sync fields between two JSONL files.
For each matching entry, if a field exists in the source file but not in the target file,
it will be added to the target file.

Usage: python sync_fields.py --source source.jsonl --target target.jsonl [--output output.jsonl]
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                sys.exit(1)
    
    return data


def find_matching_key(data):
    """Find the best key to match entries between files."""
    if not data:
        return None
    
    # Check common identifier fields in order of preference
    possible_keys = ['question_idx', 'id', 'index', 'idx', 'question_id']
    
    for key in possible_keys:
        if key in data[0]:
            # Check if this key has unique values
            values = [item.get(key) for item in data]
            if len(set(values)) == len(values) and all(v is not None for v in values):
                return key
    
    # If no good key found, use line number (index)
    return None


def sync_fields(source_file, target_file, output_file=None, matching_key=None):
    """
    Sync fields from source file to target file.
    
    Args:
        source_file (str): Path to source JSONL file
        target_file (str): Path to target JSONL file  
        output_file (str, optional): Path to output file. If None, modifies target file.
        matching_key (str, optional): Key to match entries. If None, auto-detect.
    """
    print(f"Loading source file: {source_file}")
    source_data = load_jsonl(source_file)
    
    print(f"Loading target file: {target_file}")
    target_data = load_jsonl(target_file)
    
    print(f"Source file has {len(source_data)} entries")
    print(f"Target file has {len(target_data)} entries")
    
    # Find matching key if not specified
    if matching_key is None:
        matching_key = find_matching_key(source_data)
        if matching_key:
            print(f"Auto-detected matching key: '{matching_key}'")
        else:
            print("No suitable matching key found, using line-by-line matching")
    
    # Create lookup dictionary for source data
    if matching_key:
        source_lookup = {item.get(matching_key): item for item in source_data}
        print(f"Created source lookup with {len(source_lookup)} entries")
    else:
        source_lookup = {i: item for i, item in enumerate(source_data)}
    
    # Process target data
    updated_count = 0
    field_additions = defaultdict(int)
    
    for i, target_item in enumerate(target_data):
        if matching_key:
            key_value = target_item.get(matching_key)
            source_item = source_lookup.get(key_value)
            if source_item is None:
                print(f"Warning: No matching source entry for {matching_key}={key_value}")
                continue
        else:
            if i >= len(source_data):
                print(f"Warning: Target line {i+1} has no corresponding source line")
                continue
            source_item = source_lookup[i]
        
        # Find fields in source but not in target
        added_fields = []
        for field, value in source_item.items():
            if field not in target_item:
                target_item[field] = value
                added_fields.append(field)
                field_additions[field] += 1
        
        if added_fields:
            updated_count += 1
            match_info = f"{matching_key}={target_item.get(matching_key)}" if matching_key else f"line {i+1}"
            print(f"Updated entry {match_info}: added fields {added_fields}")
    
    # Write output
    output_path = Path(output_file) if output_file else Path(target_file)
    
    print(f"\nWriting updated data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in target_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Updated {updated_count} entries")
    print(f"Field additions:")
    for field, count in sorted(field_additions.items()):
        print(f"  {field}: added to {count} entries")
    
    if output_file:
        print(f"Output written to: {output_file}")
    else:
        print(f"Target file modified in-place: {target_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync fields between two JSONL files"
    )
    parser.add_argument(
        '--source', 
        required=True,
        help='Source JSONL file path'
    )
    parser.add_argument(
        '--target', 
        required=True,
        help='Target JSONL file path (will be updated)'
    )
    parser.add_argument(
        '--output', 
        default=None,
        help='Output JSONL file path (optional, defaults to modifying target file)'
    )
    parser.add_argument(
        '--matching-key',
        default=None,
        help='Key to match entries between files (auto-detect if not specified)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.source).exists():
        print(f"Error: Source file '{args.source}' does not exist.")
        sys.exit(1)
    
    if not Path(args.target).exists():
        print(f"Error: Target file '{args.target}' does not exist.")
        sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        # Load and analyze without modifying
        source_data = load_jsonl(args.source)
        target_data = load_jsonl(args.target)
        
        matching_key = args.matching_key or find_matching_key(source_data)
        
        if matching_key:
            source_lookup = {item.get(matching_key): item for item in source_data}
        else:
            source_lookup = {i: item for i, item in enumerate(source_data)}
        
        potential_additions = defaultdict(int)
        
        for i, target_item in enumerate(target_data):
            if matching_key:
                key_value = target_item.get(matching_key)
                source_item = source_lookup.get(key_value)
                if source_item is None:
                    continue
            else:
                if i >= len(source_data):
                    continue
                source_item = source_lookup[i]
            
            for field, value in source_item.items():
                if field not in target_item:
                    potential_additions[field] += 1
        
        print(f"\n=== Dry Run Summary ===")
        print(f"Potential field additions:")
        for field, count in sorted(potential_additions.items()):
            print(f"  {field}: would be added to {count} entries")
    
    else:
        sync_fields(args.source, args.target, args.output, args.matching_key)


if __name__ == "__main__":
    main() 