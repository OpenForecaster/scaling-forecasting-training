#!/usr/bin/env python3
"""
Script to clean and filter articles based on resolution date and answer type criteria.

This script:
1. Takes an input path (directory or single file)
2. Filters articles based on:
   - Resolution date in YYYY-MM-DD format and after 2025-05-01
   - Answer type contains specific keywords
3. Saves cleaned files in a 'cleaned' subfolder
4. Logs statistics about the filtered data
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging
import argparse

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
    if not date_str:
        return False
    
    # Check basic format with regex
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return False
    
    # Try to parse the date to ensure it's a valid date
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def is_date_after_cutoff(date_str: str, cutoff_date: str = "2025-05-01") -> bool:
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
        return date_obj > cutoff_obj
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
    
    invalid_keywords = [
        "explanation", "any", "integer", "decimal", "percentage", "number", "phrase"
    ]
    
    if any(keyword in answer_type_lower for keyword in invalid_keywords):
        return False
    
    # Required keywords
    valid_keywords = [
        "title", "name",
        "month", "year", 
        # "date", 
        "location", "place",
        # "organization", "organisation", 
        "company", "corporation", "institution", "club", "team",
        "city", "town", "village", 
        "country", "nation", "state", 
        "province", "territory", # "republic", 
        # "region", "district", # "zone", "sector", "division",
        "state",  "county",  "department",
        "territory",
        "sector",
        "sport", "game", "league", "competition",  
        "award", "trophy", "medal", "prize", "reward", "honor", "recognition",
        # "tournament", "match", "athletics",
        "disease", "syndrome", "disorder", "virus", "bacteria" #  "event",
        "medical condition", "currency", "brand", 
        "venue", "team", "planet",
    ]
    
    return True # for now
    
    # Required keywords
    valid_keywords = [
        "title", "name",
        "month", "year", 
        # "date", 
        "location", "place",
        "organization", "organisation", 
        "company", "corporation", "institution", "club", "team",
        "city", "town", "village", 
        "country", "nation", "state", 
        "province", "territory", # "republic", 
        "region", "district", "zone", "sector", "division",
        "state",  "county",  "department",
        "territory",
        "sector",
        "sport", "game", "league", "competition",  
        "award", "trophy", "medal", "prize", "reward", "honor", "recognition",
        "tournament", "match", "athletics",
        "disease", "syndrome", "disorder", "virus", "bacteria", "event",
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

def load_articles_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load articles from a single JSONL file."""
    articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        article = json.loads(line.strip())
                        if article.get('question_relevant', 1) == 0:
                            continue
                        
                        articles.append(article)
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

def process_file(file_path: str, output_dir: Path) -> Dict[str, int]:
    """
    Process a single file and save cleaned version.
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing file: {file_path}")
    
    # Load articles from file
    articles = load_articles_from_file(file_path)
    if not articles:
        logger.warning(f"No articles loaded from {file_path}")
        return {"total": 0, "filtered": 0, "valid_date": 0, "valid_date_cutoff": 0, "valid_answer_type": 0}
    
    original_count = len(articles)
    valid_date_count = 0
    valid_answer_type_count = 0
    valid_date_cutoff_count = 0
    filtered_articles = []
    
    logger.info(f"Loaded {original_count} articles")
    
    # Process each article
    for article in articles:
        # Check if resolution_date exists and is valid
        resolution_date = article.get('resolution_date', '')
        
        # Skip if no resolution date or invalid format
        if not is_valid_date_format(resolution_date):
            continue
        
        valid_date_count += 1
        
        # Skip if date is not after 2025-05-01
        if not is_date_after_cutoff(resolution_date):
            continue
        
        valid_date_cutoff_count += 1
        
        # Extract question components to get answer_type
        question_title, background, resolution_criteria, answer, answer_type = extract_question_components(article)
        
        # Skip if no valid answer_type with required keywords
        if not has_valid_answer_type(answer_type):
            continue
        
        valid_answer_type_count += 1
        
        # Article passed all filters
        filtered_articles.append(article)
    
    # Create output file path
    input_file = Path(file_path)
    output_file = output_dir / f"{input_file.stem}_cleaned.jsonl"
    
    # Save filtered articles
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in filtered_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(filtered_articles)} cleaned articles to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving file {output_file}: {e}")
        return {"total": original_count, "filtered": 0, "valid_date": valid_date_count, "valid_answer_type": valid_answer_type_count}
    
    return {
        "total": original_count,
        "filtered": len(filtered_articles),
        "valid_date": valid_date_count,
        "valid_date_cutoff": valid_date_cutoff_count,
        "valid_answer_type": valid_answer_type_count
    }

def analyze_answer_types(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze answer types in the filtered articles."""
    answer_type_counts = {}
    
    for article in articles:
        question_title, background, resolution_criteria, answer, answer_type = extract_question_components(article)
        if answer_type:
            answer_type_lower = answer_type.lower()
            answer_type_counts[answer_type_lower] = answer_type_counts.get(answer_type_lower, 0) + 1
    
    return answer_type_counts

def main():
    parser = argparse.ArgumentParser(description="Clean and filter articles based on resolution date and answer type")
    
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
    
    # Create cleaned output directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        base_dir = input_path.parent
    else:
        base_dir = input_path
    
    cleaned_dir = base_dir / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    
    logger.info(f"Found {len(files_to_process)} .jsonl files to process")
    logger.info(f"Output directory: {cleaned_dir}")
    
    # Process each file
    total_stats = {
        "total": 0,
        "filtered": 0,
        "valid_date": 0,
        "valid_date_cutoff": 0,
        "valid_answer_type": 0
    }
    
    all_filtered_articles = []
    
    for file_path in files_to_process:
        try:
            stats = process_file(file_path, cleaned_dir)
            
            # Update total stats
            for key in total_stats:
                total_stats[key] += stats[key]
                
            # Load filtered articles for analysis
            input_file = Path(file_path)
            output_file = cleaned_dir / f"{input_file.stem}_cleaned.jsonl"
            if output_file.exists():
                filtered_articles = load_articles_from_file(str(output_file))
                all_filtered_articles.extend(filtered_articles)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    # Log comprehensive statistics
    logger.info("\n" + "="*60)
    logger.info("CLEANING STATISTICS")
    logger.info("="*60)
    logger.info(f"Total articles processed: {total_stats['total']}")
    logger.info(f"Articles with valid resolution dates (YYYY-MM-DD): {total_stats['valid_date']}")
    logger.info(f"Articles with resolution date after 2025-05-01: {total_stats['valid_date_cutoff']}")
    logger.info(f"Articles with valid answer types: {total_stats['valid_answer_type']}")
    logger.info(f"Final filtered articles: {total_stats['filtered']}")
    
    if total_stats['total'] > 0:
        logger.info(f"Overall filtering rate: {total_stats['filtered']/total_stats['total']*100:.1f}%")
    
    # Analyze answer types in filtered data
    if all_filtered_articles:
        answer_type_analysis = analyze_answer_types(all_filtered_articles)
        logger.info(f"\nAnswer type distribution in filtered data:")
        for answer_type, count in sorted(answer_type_analysis.items(), key=lambda x: x[1], reverse=True)[:30]:
            logger.info(f"  - {answer_type}: {count}")
    
    # Log file size information
    total_size = 0
    for file_path in cleaned_dir.glob("*_cleaned.jsonl"):
        file_size = file_path.stat().st_size
        total_size += file_size
        logger.info(f"Cleaned file: {file_path.name} ({file_size:,} bytes)")
    
    logger.info(f"Total cleaned data size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    logger.info("All files processed and cleaned!")

if __name__ == "__main__":
    main()
