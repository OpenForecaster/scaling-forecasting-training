#!/usr/bin/env python3
"""
Script to union two JSONL files based on matching question_title and answer fields.
Combines all fields from both files for matching entries.
"""

import json
import os
from typing import Dict, Any, List
from collections import defaultdict

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dictionaries.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of entries from the file
    """
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
    return entries

def create_key(entry: Dict[str, Any]) -> str:
    """
    Create a unique key for an entry based on question_title and answer.
    
    Args:
        entry (Dict[str, Any]): The entry dictionary
        
    Returns:
        str: A unique key combining question_title and answer
    """
    question_title = entry.get('question_title', '')
    answer = entry.get('answer', '')
    return f"{question_title}|{answer}"

def union_entries(entry1: Dict[str, Any], entry2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Union two entries by combining all fields from both.
    For overlapping fields, prefer non-None values and combine lists.
    
    Args:
        entry1 (Dict[str, Any]): First entry
        entry2 (Dict[str, Any]): Second entry
        
    Returns:
        Dict[str, Any]: Combined entry
    """
    unioned = {}
    
    # Get all unique keys from both entries
    all_keys = set(entry1.keys()) | set(entry2.keys())
    
    for key in all_keys:
        value1 = entry1.get(key)
        value2 = entry2.get(key)
        
        if value1 is None and value2 is not None:
            unioned[key] = value2
        elif value2 is None and value1 is not None:
            unioned[key] = value1
        elif value1 is None and value2 is None:
            unioned[key] = None
        elif isinstance(value1, list) and isinstance(value2, list):
            # Combine lists, removing duplicates while preserving order
            combined = value1.copy()
            for item in value2:
                if item not in combined:
                    combined.append(item)
            unioned[key] = combined
        elif isinstance(value1, dict) and isinstance(value2, dict):
            # Recursively union dictionaries
            unioned[key] = union_entries(value1, value2)
        else:
            # For other types, prefer the first non-None value
            unioned[key] = value1 if value1 is not None else value2
    
    return unioned

def union_files(file1_path: str, file2_path: str, output_path: str):
    """
    Union two JSONL files based on matching question_title and answer fields.
    
    Args:
        file1_path (str): Path to the first JSONL file
        file2_path (str): Path to the second JSONL file
        output_path (str): Path to save the unioned output file
    """
    print(f"Loading file 1: {file1_path}")
    entries1 = load_jsonl_file(file1_path)
    print(f"Loaded {len(entries1)} entries from file 1")
    
    print(f"Loading file 2: {file2_path}")
    entries2 = load_jsonl_file(file2_path)
    print(f"Loaded {len(entries2)} entries from file 2")
    
    # Create lookup dictionaries for both files
    lookup1 = {}
    lookup2 = {}
    
    print("Creating lookup dictionaries...")
    for entry in entries1:
        key = create_key(entry)
        if key in lookup1:
            print(f"Warning: Duplicate key in file 1: {key}")
        lookup1[key] = entry
    
    for entry in entries2:
        key = create_key(entry)
        if key in lookup2:
            print(f"Warning: Duplicate key in file 2: {key}")
        lookup2[key] = entry
    
    # Find all unique keys
    all_keys = set(lookup1.keys()) | set(lookup2.keys())
    print(f"Found {len(all_keys)} unique entries across both files")
    
    # Union only the entries that exist in both files
    unioned_entries = []
    matched_count = 0
    
    for key in all_keys:
        entry1 = lookup1.get(key)
        entry2 = lookup2.get(key)
        
        if entry1 is not None and entry2 is not None:
            # Entry exists in both files - union them
            unioned_entry = union_entries(entry1, entry2)
            unioned_entries.append(unioned_entry)
            matched_count += 1
    
    print(f"\nUnion Statistics:")
    print(f"  Entries in both files: {matched_count}")
    print(f"  Total unioned entries: {len(unioned_entries)}")
    
    # Save the unioned entries
    print(f"\nSaving unioned entries to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in unioned_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(unioned_entries)} entries to {output_path}")

def main():
    """Main function to run the union operation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Union two JSONL files based on question_title and answer")
    parser.add_argument(
        "--file1", 
        type=str, 
        default="/fast/nchandak/forecasting/datasets/synthetic/freeform/cnn_dw_forbes/with_retrieval/retrieval_20k_questions.jsonl",
        help="Path to the first JSONL file"
    )
    parser.add_argument(
        "--file2", 
        type=str, 
        default="/fast/nchandak/forecasting/datasets/synthetic/freeform/cnn_dw_forbes/combined_all_questions.jsonl",
        help="Path to the second JSONL file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/fast/nchandak/forecasting/datasets/synthetic/freeform/cnn_dw_forbes/union_questions.jsonl",
        help="Path to save the unioned output file"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.file1):
        print(f"Error: File 1 does not exist: {args.file1}")
        return 1
    
    if not os.path.exists(args.file2):
        print(f"Error: File 2 does not exist: {args.file2}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("JSONL File Union Tool")
    print("=" * 50)
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"Output: {args.output}")
    print()
    
    try:
        union_files(args.file1, args.file2, args.output)
        print("\nUnion operation completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during union operation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
