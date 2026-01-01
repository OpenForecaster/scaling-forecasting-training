#!/usr/bin/env python3
"""
Script to add news_source field from source file to target file.
Matches questions by idx (target) and qid-1 (source).
"""

import json
import argparse
from pathlib import Path
from typing import Dict


def load_source_mapping(source_file: Path) -> Dict[int, str]:
    """Load source file and create mapping from idx to news_source."""
    mapping = {}
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                qid = int(data.get('qid', 0))
                news_source = data.get('news_source')
                if news_source:
                    # Map qid-1 to idx (since qid is 1-based and idx is 0-based)
                    idx = qid - 1
                    mapping[idx] = news_source
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to parse line in source file: {e}")
                continue
    return mapping


def add_news_source_to_target(target_file: Path, source_file: Path, output_file: Path = None):
    """Add news_source field from source file to target file."""
    # Load source mapping
    source_mapping = load_source_mapping(source_file)
    print(f"Loaded {len(source_mapping)} news_source mappings from source file")
    
    # Determine output path
    if output_file is None:
        output_file = target_file
    
    # Process target file
    processed_count = 0
    missing_count = 0
    updated_questions = []
    
    with open(target_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                question = json.loads(line)
                idx = question.get('idx')
                
                if idx is not None and idx in source_mapping:
                    question['news_source'] = source_mapping[idx]
                    processed_count += 1
                else:
                    # If no match found, set to None or empty string
                    question['news_source'] = None
                    missing_count += 1
                    if idx is not None:
                        print(f"Warning: No news_source found for idx={idx} (line {line_num})")
                
                updated_questions.append(question)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in target file: {e}")
                continue
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for question in updated_questions:
            f.write(json.dumps(question) + '\n')
    
    print(f"Processed {processed_count} questions with news_source")
    print(f"Missing news_source for {missing_count} questions")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add news_source field from source file to target file"
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default="/fast/nchandak/forecasting/evals/freeform/manual/news5-retrieval_1000/fix/grok-4.1-fast:online_eval_size_1000_generations_5_date.jsonl",
        help="Path to target JSONL file"
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default="/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_1000_30.jsonl",
        help="Path to source JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: overwrites target file)"
    )
    
    args = parser.parse_args()
    
    target_path = Path(args.target_file)
    source_path = Path(args.source_file)
    
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    add_news_source_to_target(target_path, source_path, Path(args.output_file) if args.output_file else None)


if __name__ == "__main__":
    main()

