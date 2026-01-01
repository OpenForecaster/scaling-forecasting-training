#!/usr/bin/env python3
"""
Script to extract and remove start dates from question backgrounds.

This script:
1. Takes input as either a directory (processes all .jsonl files) or a single .jsonl file
2. For each entry, examines the background field to find start date information
3. Extracts the start date and removes it from the background
4. Adds the extracted date to a new field 'question_start_date'
5. Saves the modified entries back to the same files
"""

import os
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

def extract_and_remove_start_date(background: str) -> Tuple[str, str]:
    """
    Extract start date from background text and remove it.
    
    Args:
        background: The background text to process
        
    Returns:
        Tuple of (cleaned_background, extracted_start_date)
    """
    if not background:
        return background, ""
    
    # Define patterns for start date extraction
    start_date_patterns = [
        # "Question Start Date: 10th March 2024" or "Question Start Date: March 10, 2024"
        r'Question Start Date:\s*([^.]+?)(?:\.|$)',
        # "Start Date: 2024-03-10" or "Start Date: March 10, 2024"
        r'Start Date:\s*([^.]+?)(?:\.|$)',
        # "Question start date: ..." (case insensitive)
        r'(?i)question start date:\s*([^.]+?)(?:\.|$)',
        # "Start date: ..." (case insensitive)
        r'(?i)start date:\s*([^.]+?)(?:\.|$)',
        # "The question starts on [date]"
        r'(?i)the question starts on\s+([^.]+?)(?:\.|$)',
        # "Starting from [date]"
        r'(?i)starting from\s+([^.]+?)(?:\.|$)',
        # "As of [date]" at the beginning
        r'(?i)^as of\s+([^.]+?)(?:\.|,)',
    ]
    
    extracted_date = ""
    
    # Try each pattern to find and extract start date
    for pattern in start_date_patterns:
        match = re.search(pattern, background)
        if match:
            # Extract the date portion
            date_text = match.group(1).strip()
            
            # Clean up common suffixes and prefixes
            date_text = re.sub(r'\s*[\.,;]\s*$', '', date_text)  # Remove trailing punctuation
            date_text = re.sub(r'^\s*[\.,;]\s*', '', date_text)  # Remove leading punctuation
            
            # Try to normalize the date format
            normalized_date = normalize_date_format(date_text)
            if normalized_date:
                extracted_date = normalized_date
            else:
                extracted_date = date_text
            
            break  # Stop after first match
    
    return extracted_date

def normalize_date_format(date_text: str) -> str:
    """
    Try to normalize various date formats to a consistent format.
    
    Args:
        date_text: Raw date text extracted from background
        
    Returns:
        Normalized date string or empty string if can't parse
    """
    if not date_text:
        return ""
    
    # Remove common prefixes/suffixes
    date_text = re.sub(r'^\s*(on\s+|the\s+)', '', date_text, flags=re.IGNORECASE)
    date_text = re.sub(r'\s*(onwards?|forward)\s*$', '', date_text, flags=re.IGNORECASE)
    
    # Common date patterns to recognize and normalize
    date_patterns = [
        # YYYY-MM-DD format
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3'),
        # DD/MM/YYYY or MM/DD/YYYY
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\2-\1'),
        # DD.MM.YYYY
        (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', r'\3-\2-\1'),
        # Month DD, YYYY (e.g., "March 10, 2024")
        (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', lambda m: format_month_day_year(m.group(1), m.group(2), m.group(3))),
        # DD Month YYYY (e.g., "10 March 2024" or "10th March 2024")
        (r'(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})', lambda m: format_day_month_year(m.group(1), m.group(2), m.group(3))),
        # Month YYYY (e.g., "March 2024")
        (r'^(\w+)\s+(\d{4})$', lambda m: format_month_year(m.group(1), m.group(2))),
    ]
    
    for pattern, replacement in date_patterns:
        if callable(replacement):
            match = re.search(pattern, date_text, re.IGNORECASE)
            if match:
                result = replacement(match)
                if result:
                    return result
        else:
            if re.search(pattern, date_text):
                return re.sub(pattern, replacement, date_text)
    
    # If no pattern matches, return the original text (might be a valid date description)
    return date_text.strip()

def format_month_day_year(month_str: str, day_str: str, year_str: str) -> str:
    """Convert 'Month DD, YYYY' to 'YYYY-MM-DD' format."""
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
        'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    month_num = month_map.get(month_str.lower())
    if month_num:
        day_num = day_str.zfill(2)
        return f"{year_str}-{month_num}-{day_num}"
    return ""

def format_day_month_year(day_str: str, month_str: str, year_str: str) -> str:
    """Convert 'DD Month YYYY' to 'YYYY-MM-DD' format."""
    return format_month_day_year(month_str, day_str, year_str)

def format_month_year(month_str: str, year_str: str) -> str:
    """Convert 'Month YYYY' to 'YYYY-MM' format."""
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
        'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    month_num = month_map.get(month_str.lower())
    if month_num:
        return f"{year_str}-{month_num}"
    return ""

def extract_question_components(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract question components from entry.
    Returns (question_title, background, resolution_criteria)
    """
    # Try to get from individual fields first
    question_title = entry.get('question_title', '')
    background = entry.get('background', '')
    resolution_criteria = entry.get('resolution_criteria', '')
    
    # If we have these fields, return them
    if question_title and background and resolution_criteria:
        return question_title, background, resolution_criteria
    
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
    
    return question_title, background, resolution_criteria

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

def get_files_to_process(input_path: str) -> List[str]:
    """Get list of .jsonl files to process."""
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix == '.jsonl':
            return [str(path)]
        else:
            logger.error(f"Input file must have .jsonl extension: {input_path}")
            return []
    elif path.is_dir():
        jsonl_files = list(path.glob('*.jsonl'))
        return [str(f) for f in jsonl_files]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return []
    
    
def get_date_from_background(background: str) -> str:
    """Get the date from the background."""
    if not background:
        return -1
    
    years = re.findall(r'\b\d{4}\b', background)
    if "start date" not in background.lower():
        return -1 # no start date
    
    # return the position at the end of the first occurrence of year (if exists)
    for year in years:
        if year in background:
            return background.index(year) + len(year)
        
    return -1

def process_file(file_path: str) -> None:
    """Process a single file for start date extraction and removal."""
    logger.info(f"Processing file: {file_path}")
    
    # Load articles from file
    articles = load_articles_from_file(file_path)
    if not articles:
        logger.warning(f"No articles loaded from {file_path}")
        return
    
    original_count = len(articles)
    processed_count = 0
    extracted_count = 0
    
    logger.info(f"Loaded {original_count} articles")
    
    # Process each article
    for article in articles:
        # Extract question components
        
        
        question_title, background, resolution_criteria = extract_question_components(article)
        
        if not background:
            continue
        
        if background.lower()[:2] == ". ":
            background = background[2:]
            article['background'] = background
        
        start_date_index = get_date_from_background(background)
        
        if start_date_index <= 0:
            continue
        
        start_date_ending_index = background.find(".", start_date_index)
        start_date_str = background[:start_date_ending_index]
        cleaned_background = background[start_date_ending_index:]
        
        # Extract and remove start date from background
        start_date = extract_and_remove_start_date(start_date_str)
        
        # Update the entry if we found a start date or if background was cleaned
        if start_date or cleaned_background.lower() != background.lower() :
            processed_count += 1
            
            # Update background in the appropriate field
            if 'background' in article:
                article['background'] = cleaned_background
            
            # # Also update in final_question if that's where it came from
            # if 'final_question' in article and not article.get('background'):
            #     # Update the background tag in final_question
            #     final_question = article['final_question']
            #     # Replace the background tag content
            #     background_pattern = r'(<background>)(.*?)(</background>)'
            #     if re.search(background_pattern, final_question, re.DOTALL):
            #         updated_final_question = re.sub(
            #             background_pattern,
            #             rf'\1{cleaned_background}\3',
            #             final_question,
            #             flags=re.DOTALL
            #         )
            #         article['final_question'] = updated_final_question
            
            # Add start date field if we extracted one
            if start_date:
                article['question_start_date'] = start_date
                extracted_count += 1
    
    logger.info(f"Processed {processed_count} articles, extracted start dates from {extracted_count}")
    
    # Save results back to the same file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Updated {file_path} - {extracted_count} articles now have question_start_date field")
        
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract and remove start dates from question backgrounds")
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input directory (processes all .jsonl files) or single .jsonl file"
    )
    
    args = parser.parse_args()
    
    # Get files to process
    files_to_process = get_files_to_process(args.input_path)
    
    if not files_to_process:
        logger.error("No .jsonl files found to process")
        return
    
    logger.info(f"Found {len(files_to_process)} .jsonl files to process")
    
    # Process each file
    total_processed = 0
    total_extracted = 0
    
    for file_path in files_to_process:
        try:
            process_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("All files processed!")

if __name__ == "__main__":
    main()
