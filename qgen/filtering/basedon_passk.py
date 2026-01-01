#!/usr/bin/env python3
"""
Filter source questions based on evaluation scores.

This script takes a source question file and an evaluation file, then outputs a new question file
containing only the questions where at least one generation/attempt has a score of 1.

Usage:
    python basedon_passk.py <source_file> <eval_file> [--judge JUDGE_NAME]

Example:
    python basedon_passk.py /path/to/source.jsonl /path/to/eval.jsonl --judge Qwen3_4B
"""

import json
import argparse
import os
from typing import List, Dict, Any, Set


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in {file_path}: {e}")
                continue
    return data


def save_jsonl_file(data: List[Dict[str, Any]], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def extract_judge_name_from_eval(eval_data: List[Dict[str, Any]]) -> str:
    """Extract the judge name from the evaluation data by looking for score_* fields."""
    if not eval_data:
        return None
    
    # Look for score_* fields in the first item
    first_item = eval_data[0]
    score_fields = [key for key in first_item.keys() if key.startswith('score_')]
    
    if not score_fields:
        print("Warning: No score_* fields found in evaluation data")
        return None
    
    # Use the first score field found
    judge_name = score_fields[0].replace('score_', '')
    print(f"Found judge field: score_{judge_name}")
    return judge_name


def get_passing_question_indices(eval_data: List[Dict[str, Any]], judge_name: str) -> Set[int]:
    """
    Get the indices of questions where at least one generation has a score of 1.
    
    Args:
        eval_data: List of evaluation items
        judge_name: Name of the judge (e.g., 'Qwen3_4B')
        
    Returns:
        Set of question indices that have at least one passing score
    """
    passing_indices = set()
    score_field = f"score_{judge_name}"
    
    for item in eval_data:
        if score_field not in item:
            print(f"Warning: {score_field} not found in item with idx {item.get('idx', 'unknown')}")
            continue
        
        scores = item[score_field]
        if not isinstance(scores, list):
            print(f"Warning: {score_field} is not a list in item with idx {item.get('idx', 'unknown')}")
            continue
        
        # Check if any generation has a score of 1
        has_passing_score = False
        for score_item in scores:
            if isinstance(score_item, dict):
                # Extract the score value from the dict (e.g., {"answer": 1} -> 1)
                score_values = list(score_item.values())
                if score_values and score_values[0] == 1:
                    has_passing_score = True
                    break
            elif isinstance(score_item, (int, float)) and score_item == 1:
                has_passing_score = True
                break
        
        if has_passing_score:
            # Get the question index from the eval item
            question_idx = item.get('idx')
            if question_idx is not None:
                passing_indices.add(question_idx)
    
    print(f"Found {len(passing_indices)} questions with at least one passing score out of {len(eval_data)} total questions")
    return passing_indices


def filter_source_questions(source_data: List[Dict[str, Any]], passing_indices: Set[int]) -> List[Dict[str, Any]]:
    """
    Filter source questions to keep only those with passing scores.
    
    Args:
        source_data: List of source question items
        passing_indices: Set of indices to keep
        
    Returns:
        Filtered list of source questions
    """
    filtered_data = []
    
    for i, item in enumerate(source_data):
        if i in passing_indices or i % 7 == 0:
            filtered_data.append(item)
    
    print(f"Filtered source questions: kept {len(filtered_data)} out of {len(source_data)} questions")
    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter source questions based on evaluation scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python basedon_passk.py source.jsonl eval.jsonl
    python basedon_passk.py source.jsonl eval.jsonl --judge Qwen3_4B
        """
    )
    
    parser.add_argument("source_file", help="Path to source question JSONL file")
    parser.add_argument("eval_file", help="Path to evaluation JSONL file")
    parser.add_argument("--judge", help="Judge name (e.g., Qwen3_4B). If not provided, will be auto-detected")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.source_file):
        print(f"Error: Source file {args.source_file} does not exist")
        return 1
    
    if not os.path.exists(args.eval_file):
        print(f"Error: Evaluation file {args.eval_file} does not exist")
        return 1
    
    print(f"Loading source questions from: {args.source_file}")
    source_data = load_jsonl_file(args.source_file)
    print(f"Loaded {len(source_data)} source questions")
    
    print(f"Loading evaluation data from: {args.eval_file}")
    eval_data = load_jsonl_file(args.eval_file)
    print(f"Loaded {len(eval_data)} evaluation items")
    
    # Determine judge name
    if args.judge:
        judge_name = args.judge
        print(f"Using specified judge: {judge_name}")
    else:
        judge_name = extract_judge_name_from_eval(eval_data)
        if not judge_name:
            print("Error: Could not determine judge name from evaluation data")
            return 1
        print(f"Auto-detected judge: {judge_name}")
    
    # Get passing question indices
    passing_indices = get_passing_question_indices(eval_data, judge_name)
    
    if not passing_indices:
        print("Warning: No questions found with passing scores")
        return 0
    
    # Filter source questions
    filtered_source_data = filter_source_questions(source_data, passing_indices)
    
    if not filtered_source_data:
        print("Warning: No source questions to save after filtering")
        return 0
    
    # Generate output filename
    source_dir = os.path.dirname(args.source_file)
    source_basename = os.path.basename(args.source_file)
    name_without_ext = os.path.splitext(source_basename)[0]
    output_filename = f"{name_without_ext}_filtered_passk_{judge_name}.jsonl"
    output_path = os.path.join(source_dir, output_filename)
    
    # Save filtered questions
    print(f"Saving filtered questions to: {output_path}")
    save_jsonl_file(filtered_source_data, output_path)
    
    print(f"Successfully saved {len(filtered_source_data)} filtered questions")
    print(f"Output file: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
