#!/usr/bin/env python3
"""
Script to compare resolution dates with model-predicted dates.

For each question:
1. Collect correct answers based on judge scores
2. Extract dates (epoch seconds) from correct answers
3. Calculate median date from correct answers
4. Compare original resolution_date with median_date + 24hrs
5. Add resolution_date_correct field (1 if within 24hrs, 0 if no correct answers)
"""

import json
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def get_correct_answer_indices(score_field: List[Dict[str, float]]) -> List[int]:
    """Get indices of responses with correct scores (score == 1.0)."""
    correct_indices = []
    for idx, score_dict in enumerate(score_field):
        # Check if any answer in this response has score 1.0
        if any(score == 1.0 for score in score_dict.values()):
            correct_indices.append(idx)
    return correct_indices


def extract_dates_from_correct_answers(
    extracted_answer: List[Dict[str, Any]],
    correct_indices: List[int]
) -> List[int]:
    """Extract epoch seconds dates from correct answer responses."""
    dates = []
    for idx in correct_indices:
        if idx < len(extracted_answer):
            # Each extracted_answer entry is a dict mapping answer to epoch seconds
            # Collect all dates from this response, filtering out None values
            for date_epoch in extracted_answer[idx].values():
                if date_epoch is not None:
                    dates.append(date_epoch)
    return dates


def process_question(question: Dict[str, Any], judge: str, update_date_threshold: Optional[int] = None) -> Dict[str, Any]:
    """Process a single question and add resolution_date_correct field.
    
    Args:
        question: Question dictionary
        judge: Judge name
        update_date_threshold: If provided, update resolution_date with median_date for questions with majority correct
    
    Returns:
        Processed question dictionary with resolution_date_correct field and optionally updated resolution_date
    """
    score_field = f"score_{judge}"
    
    # assume false by default
    question["has_majority_correct"] = False
    question["resolution_date_correct"] = 0
    
    # Check if score field exists
    if score_field not in question:
        return question
    
    score_data = question[score_field]
    if not isinstance(score_data, list):
        return question
    
    # Get indices of correct answers
    correct_indices = get_correct_answer_indices(score_data)
    
    # If no correct answers, set field to 0
    if not correct_indices:
        return question
    
    # Only process if at least half of the responses are correct
    total_responses = len(score_data)
    num_correct = len(correct_indices)
    has_majority = num_correct * 2 >= total_responses
    question["has_majority_correct"] = has_majority
    
    if not has_majority:
        return question
    
    # Extract dates from correct answers
    if "extracted_answer" not in question:
        return question
    
    extracted_answer = question["extracted_answer"]
    if not isinstance(extracted_answer, list):
        return question
    
    dates = extract_dates_from_correct_answers(extracted_answer, correct_indices)
    
    if not dates:
        return question
    
    # Sort dates and get median
    dates_sorted = sorted(dates)
    median_date = statistics.median(dates_sorted)
    median_date = min(dates_sorted)
    question["median_date"] = median_date
    
    # Get original resolution_date
    if "resolution_date" not in question:
        return question
    
    resolution_date = question["resolution_date"]
    
    # If update_date_threshold is provided, update resolution_date with median_date
    if update_date_threshold is not None:
        question["resolution_date"] = int(median_date)
        resolution_date = int(median_date)
    
    # Check if resolution_date <= median_date + 24hrs (86400 seconds)
    # Add 24 hours (86400 seconds) to median_date
    median_date_plus_24hrs = median_date + 86400*1
    
    if resolution_date <= median_date_plus_24hrs:
        question["resolution_date_correct"] = 1
    else:
        question["resolution_date_correct"] = 0
    
    return question


def main():
    parser = argparse.ArgumentParser(
        description="Compare resolution dates with model-predicted dates"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/fast/nchandak/forecasting/evals/freeform/manual/validation-retrieval_207/grok-4.1-fast:online_eval_size_207_generations_5_datefixed.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="Qwen3_4B",
        help="Judge name (e.g., Qwen3_4B, Llama_4_Scout)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: overwrites input file)"
    )
    parser.add_argument(
        "--update_date",
        type=str,
        default=None,
        help="Date in YYYYMMDD format. If provided, updates resolution_date with median_date and filters questions"
    )
    
    args = parser.parse_args()
    
    # Parse update_date if provided
    update_date_epoch = None
    if args.update_date:
        try:
            # Parse YYYYMMDD format
            date_obj = datetime.strptime(args.update_date, "%Y%m%d")
            update_date_epoch = int(date_obj.timestamp())
        except ValueError:
            raise ValueError(f"Invalid date format: {args.update_date}. Expected YYYYMMDD format.")
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path
    
    # Process each line in the JSONL file
    processed_questions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                question = json.loads(line)
                processed_question = process_question(question, args.judge, update_date_threshold=update_date_epoch)
                processed_questions.append(processed_question)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    # Add model_filter field if update_date is provided
    if update_date_epoch is not None:
        delta = 86400 * 1
        kept_count = 0
        for question in processed_questions:
            # Check if question should be kept
            has_majority = question.get("has_majority_correct", False)
            resolution_date = question.get("resolution_date")
            
            if has_majority and resolution_date is not None and resolution_date >= update_date_epoch - delta:
                question["model_filter"] = 1
                kept_count += 1
            else:
                question["model_filter"] = 0
        
        print(f"Added model_filter field: {kept_count} questions with model_filter=1, {len(processed_questions) - kept_count} with model_filter=0")
    else:
        # If update_date is not provided, set model_filter to 0 for all questions
        for question in processed_questions:
            question["model_filter"] = 0
    
    # Clean up temporary fields before writing
    # for question in processed_questions:
    #     question.pop("has_majority_correct", None)
    #     question.pop("median_date", None)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in processed_questions:
            f.write(json.dumps(question) + '\n')
    
    print(f"Processed {len(processed_questions)} questions")
    print(f"Output written to: {output_path}")
    
    # Print summary statistics
    correct_count = sum(1 for q in processed_questions if q.get("resolution_date_correct", 0) == 1)
    no_correct_answers = sum(1 for q in processed_questions if q.get("resolution_date_correct", 0) == 0 and q.get("has_majority_correct", 0) == 1)
    print(f"Questions with resolution_date_correct=1: {correct_count}")
    print(f"Questions with resolution_date_correct=0: {no_correct_answers}")


if __name__ == "__main__":
    main()

