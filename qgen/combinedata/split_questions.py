#!/usr/bin/env python3
"""
Script to split questions into training and validation sets based on resolution date.

This script:
1. Takes an input JSONL file (default: combined_all_questions.jsonl)
2. Filters items based on:
   - Valid resolution date format (YYYY-MM-DD)
   - Answer type contains specific keywords (excluding invalid ones)
3. Sorts remaining items by resolution_date
4. Creates training set (first 90%) and validation set (last 10%)
   - Validation set items resolve after training set items
5. Saves both sets to separate files
"""

import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def is_valid_date_format(date_str: str) -> bool:
    """
    Check if date string is in YYYY-MM-DD format.
    
    Args:
        date_str: Date string to validate
    Returns:
        True if valid YYYY-MM-DD format, False otherwise
    """
    # Ensure input is a string
    if not isinstance(date_str, str) or not date_str:
        return False

    # Check basic format with regex
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return False

    # Try to parse the date to ensure it's a valid date
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except Exception:
        return False


def is_date_after_cutoff(date_str: str, cutoff_date: str = "2020-01-01") -> bool:
    """
    Check if date is after the cutoff date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        cutoff_date: Cutoff date in YYYY-MM-DD format
        
    Returns:
        True if date is after cutoff, False otherwise
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        cutoff_obj = datetime.strptime(cutoff_date, '%Y-%m-%d')
        high_cutoff_obj = datetime.strptime("2030-01-01", '%Y-%m-%d')
        high_cutoff_obj = datetime.strptime("2025-02-01", '%Y-%m-%d')
        return date_obj > cutoff_obj and date_obj < high_cutoff_obj
    except ValueError:
        return False

def has_valid_answer_type(answer_type: str) -> bool:
    """
    Check if answer_type contains any of the required keywords.
    
    Args:
        answer_type: Answer type string to check
        
    Returns:
        True if contains valid keywords, False otherwise
    """
    if not answer_type:
        return False
    
    # Convert to lowercase for case-insensitive matching
    answer_type_lower = answer_type.lower()
    
    # Invalid keywords - exclude these
    invalid_keywords = [
        "explanation", "any", "integer", "decimal", "percentage", "range", "phrase",
    ]
    
    if any(keyword in answer_type_lower for keyword in invalid_keywords):
        return False
    
    # Required keywords - must contain at least one of these
    valid_keywords = [
        "title", "name",
        "month", "year", 
        "binary",
        # "date", 
        "location", "place",
        "color", "colour",
        "platform", "device", "system", "operating system", "OS",
        "zodiac sign",
        "political party", "profession", "occupation", "job", "position", "discipline",
        "organization", "organisation", 
        "company", "corporation", "institution", "club", "team",
        "city", "town", "village", 
        "country", "nation", "state", 
        "province", "territory", # "republic", 
        "district", # "region", "zone", "sector", "division",
        "state",  "county",  "department",
        "territory",
        "sector",
        "sport", "game", "league", "competition",  
        "award", "trophy", "medal", "prize", "reward", "honor", "recognition",
        "tournament", # "match", "athletics",
        "disease", "syndrome", "disorder", "virus", "bacteria" #  "event",
        "medical condition", "currency", "brand", 
        "venue", "team", "planet",
    ]
    
    
    # Check if any keyword exists in answer_type
    return any(keyword in answer_type_lower for keyword in valid_keywords)

def extract_question_components(entry: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    """
    Extract question components from entry.
    Returns (question_title, background, resolution_criteria, answer, answer_type)
    """
    # Try to get from individual fields first
    question_title = entry.get('question_title', '')
    background = entry.get('background', '')
    resolution_criteria = entry.get('resolution_criteria', '')
    answer = entry.get('answer', '')
    answer_type = entry.get('answer_type', '')
    
    # If we have these fields, return them
    if question_title and answer_type:
        return question_title, background, resolution_criteria, answer, answer_type
    
    # Otherwise, try to extract from final_question field
    final_question = entry.get('final_question', '')
    if final_question:
        def extract_tag_content(text: str, tag: str) -> str:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            last_open = text.rfind(open_tag)
            if last_open == -1:
                return ""
            start = last_open + len(open_tag)
            end = text.find(close_tag, start)
            if end == -1:
                return ""
            return text[start:end].strip()
        
        if not question_title:
            question_title = extract_tag_content(final_question, 'question_title')
        if not background:
            background = extract_tag_content(final_question, 'background')
        if not resolution_criteria:
            resolution_criteria = extract_tag_content(final_question, 'resolution_criteria')
        if not answer:
            answer = extract_tag_content(final_question, 'answer')
        if not answer_type:
            answer_type = extract_tag_content(final_question, 'answer_type')
    
    return question_title, background, resolution_criteria, answer, answer_type

def load_questions_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load questions from a JSONL file."""
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        line_data = json.loads(line.strip())
                        # if date is in ISO format, convert to YYYY-MM-DD
                        if 'resolution_date' in line_data:
                            if isinstance(line_data['resolution_date'], int):
                                line_data["resolution_date"] = datetime.fromtimestamp(line_data['resolution_date']).strftime('%Y-%m-%d')
                        
                        if 'question_start_date' in line_data:
                            if isinstance(line_data['question_start_date'], int):
                                line_data['question_start_date'] = datetime.fromtimestamp(line_data['question_start_date']).strftime('%Y-%m-%d')
                        
                        questions.append(line_data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    return questions

def filter_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter questions based on resolution date and answer type criteria.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        List of filtered questions
    """
    filtered_questions = []
    total_count = len(questions)
    valid_date_count = 0
    valid_answer_type_count = 0
    
    logger.info(f"Filtering {total_count} questions...")
    
    for question in questions:
        # Check if resolution_date exists and is valid
        resolution_date = question.get('resolution_date', '')
        
        # Skip if no resolution date or invalid format
        if not is_valid_date_format(resolution_date):
            continue
        
        if not is_date_after_cutoff(resolution_date):
            continue
        
        valid_date_count += 1
        
        # Extract question components to get answer_type
        question_title, background, resolution_criteria, answer, answer_type = extract_question_components(question)
        
        # Skip if no valid answer_type with required keywords
        if not has_valid_answer_type(answer_type):
            continue
        
        valid_answer_type_count += 1
        
        # Question passed all filters
        filtered_questions.append(question)
    
    logger.info(f"Filtering results:")
    logger.info(f"  - Total questions: {total_count}")
    logger.info(f"  - Valid resolution dates: {valid_date_count}")
    logger.info(f"  - Valid answer types: {valid_answer_type_count}")
    logger.info(f"  - Final filtered questions: {len(filtered_questions)}")
    
    return filtered_questions

def split_questions_by_date(questions: List[Dict[str, Any]], train_ratio: float = 0.9) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split questions into training and validation sets based on resolution date.
    
    Args:
        questions: List of filtered questions
        train_ratio: Ratio of questions to use for training (default: 0.9)
        
    Returns:
        Tuple of (training_questions, validation_questions)
    """
    # Sort questions by resolution_date
    sorted_questions = sorted(questions, key=lambda x: x.get('resolution_date', ''))
    
    # Calculate split point
    split_index = int(len(sorted_questions) * train_ratio)
    
    training_questions = sorted_questions[:split_index]
    validation_questions = sorted_questions[split_index:]
    
    logger.info(f"Splitting {len(sorted_questions)} questions:")
    logger.info(f"  - Training set: {len(training_questions)} questions")
    logger.info(f"  - Validation set: {len(validation_questions)} questions")
    
    # Log date ranges
    if training_questions:
        train_start = training_questions[0].get('resolution_date', '')
        train_end = training_questions[-1].get('resolution_date', '')
        logger.info(f"  - Training date range: {train_start} to {train_end}")
    
    if validation_questions:
        val_start = validation_questions[0].get('resolution_date', '')
        val_end = validation_questions[-1].get('resolution_date', '')
        logger.info(f"  - Validation date range: {val_start} to {val_end}")
    
    return training_questions, validation_questions

def save_questions_to_file(questions: List[Dict[str, Any]], output_path: str):
    """Save questions to a JSONL file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for question in questions:
                f.write(json.dumps(question, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(questions)} questions to {output_path}")
    except Exception as e:
        logger.error(f"Error saving file {output_path}: {e}")

def analyze_answer_types(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze answer types in the questions."""
    answer_type_counts = {}
    
    for question in questions:
        question_title, background, resolution_criteria, answer, answer_type = extract_question_components(question)
        if answer_type:
            answer_type_lower = answer_type.lower()
            answer_type_counts[answer_type_lower] = answer_type_counts.get(answer_type_lower, 0) + 1
    
    return answer_type_counts

def main():
    parser = argparse.ArgumentParser(description="Split questions into training and validation sets based on resolution date")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="/fast/nchandak/forecasting/datasets/synthetic/freeform/cnn_dw_forbes/combined_all_questions.jsonl",
        help="Path to input JSONL file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as input file)"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.98,
        help="Ratio of questions to use for training (default: 0.9)"
    )
    
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    # Load questions
    logger.info(f"Loading questions from: {args.input_file}")
    questions = load_questions_from_file(args.input_file)
    
    if not questions:
        logger.error("No questions loaded from input file")
        return
    
    # Filter questions
    filtered_questions = filter_questions(questions)
    
    if not filtered_questions:
        logger.error("No questions passed the filtering criteria")
        return
    
    # Split questions
    training_questions, validation_questions = split_questions_by_date(filtered_questions, args.train_ratio)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be created")
        
        # Analyze answer types
        train_answer_types = analyze_answer_types(training_questions)
        val_answer_types = analyze_answer_types(validation_questions)
        
        logger.info(f"\nTraining set answer type distribution:")
        for answer_type, count in sorted(train_answer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {answer_type}: {count}")
        
        logger.info(f"\nValidation set answer type distribution:")
        for answer_type, count in sorted(val_answer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {answer_type}: {count}")
        
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input_file).parent
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filenames
    input_stem = Path(args.input_file).stem
    train_output = output_dir / f"{input_stem}_train.jsonl"
    val_output = output_dir / f"{input_stem}_validation.jsonl"
    
    # Save files
    logger.info(f"Saving split files to: {output_dir}")
    save_questions_to_file(training_questions, str(train_output))
    save_questions_to_file(validation_questions, str(val_output))
    
    # Analyze and log answer type distribution
    train_answer_types = analyze_answer_types(training_questions)
    val_answer_types = analyze_answer_types(validation_questions)
    
    logger.info(f"\nFinal answer type distribution:")
    logger.info(f"Training set top answer types:")
    for answer_type, count in sorted(train_answer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  - {answer_type}: {count}")
    
    logger.info(f"Validation set top answer types:")
    for answer_type, count in sorted(val_answer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  - {answer_type}: {count}")
    
    logger.info(f"\nSplit completed successfully!")
    logger.info(f"Training file: {train_output}")
    logger.info(f"Validation file: {val_output}")

if __name__ == "__main__":
    main()
