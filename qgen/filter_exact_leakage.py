#!/usr/bin/env python3
"""
Script to filter out entries with exact leakage in question components.

This script:
1. Takes a single .jsonl file as input
2. For each entry, checks if the "answer" appears in question_title, resolution_criteria, answer_type, or background
3. Keeps only entries where the answer is NOT present in these fields
4. Saves the filtered entries to a new file with an extra suffix
"""

import argparse
import logging
import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_question_components(entry: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Extract question components from entry.
    Returns (question_title, background, resolution_criteria, answer_type)
    """
    # Try to get from individual fields first
    question_title = entry.get('question_title', '')
    background = entry.get('background', '')
    resolution_criteria = entry.get('resolution_criteria', '')
    answer_type = entry.get('answer_type', '')
    
    # If we have these fields, return them
    if question_title and background and resolution_criteria:
        return question_title, background, resolution_criteria, answer_type
    
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
        if not answer_type:
            answer_type = extract_tag_content(final_question, 'answer_type')
    
    return question_title, background, resolution_criteria, answer_type

def has_answer_leakage(entry: Dict[str, Any]) -> bool:
    """
    Check if the answer appears in any of the question components.
    
    Args:
        entry: The JSON entry to check
        
    Returns:
        True if answer is found in question components (leakage detected), False otherwise
    """
    answer = entry.get('answer', '')
    if not answer or not answer.strip():
        return False
    
    # Extract question components
    question_title, background, resolution_criteria, answer_type = extract_question_components(entry)
    
    # Check if answer appears in any of the components
    components_to_check = [
        question_title,
        background, 
        resolution_criteria,
        answer_type
    ]
    
    # Clean the answer for comparison (strip whitespace, convert to lowercase)
    clean_answer = answer.strip().lower()
    
    for component in components_to_check:
        if component and clean_answer in component.lower():
            return True
    
    return False

def load_articles_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load articles from a single JSONL file."""
    articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        articles.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    return articles

def filter_file(input_file: str, output_file: str) -> None:
    """Filter a single file to remove entries with answer leakage."""
    logger.info(f"Processing file: {input_file}")
    
    # Load articles from file
    articles = load_articles_from_file(input_file)
    if not articles:
        logger.warning(f"No articles loaded from {input_file}")
        return
    
    original_count = len(articles)
    filtered_articles = []
    leakage_count = 0
    
    logger.info(f"Loaded {original_count} articles")
    
    # Process each article
    for i, article in enumerate(articles):
        if has_answer_leakage(article):
            leakage_count += 1
            logger.debug(f"Entry {i+1} has leakage - removing")
        else:
            filtered_articles.append(article)
    
    logger.info(f"Filtered out {leakage_count} entries with leakage")
    logger.info(f"Kept {len(filtered_articles)} entries without leakage")
    
    # Save filtered results to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in filtered_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Filtered results saved to {output_file}")
        logger.info(f"Removed {leakage_count}/{original_count} entries ({leakage_count/original_count*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error saving file {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Filter out entries with exact answer leakage")
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input .jsonl file"
    )
    
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_filtered",
        help="Suffix to add to output filename (default: _filtered)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    if not input_path.suffix == '.jsonl':
        logger.error(f"Input file must have .jsonl extension: {args.input_file}")
        return
    
    # Generate output filename
    output_file = str(input_path.with_suffix('')) + args.output_suffix + '.jsonl'
    
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Process the file
    try:
        filter_file(args.input_file, output_file)
    except Exception as e:
        logger.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
