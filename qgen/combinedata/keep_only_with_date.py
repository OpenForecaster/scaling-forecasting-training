#!/usr/bin/env python3
"""
Script to convert a single .jsonl file from news question generation format to standardized format.

This script:
1. Takes an input .jsonl file path as argument
2. Processes entries using the same logic as combine_and_split.py
3. Outputs converted file with _converted suffix in the same directory
"""

import os
import json
import argparse
from typing import Dict, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_question_components(final_question: str) -> Dict[str, str]:
    """
    Extract question components from final_question field.
    Returns dict with question_title, background, resolution_criteria, answer, answer_type
    """
    def extract_last_tag_block(text: str, tag: str) -> str:
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

    tags = ["question_title", "background", "resolution_criteria", "answer", "answer_type"]
    result = {tag: extract_last_tag_block(final_question, tag) for tag in tags}
    return result

def is_valid_news_entry(entry: Dict[str, Any]) -> bool:
    """
    Check if a news entry meets our filtering criteria.
    """
    
    # Check validity flags
    if 'final_question_valid' in entry and int(entry['final_question_valid']) != 1:
        return False
    
    if 'no_good_question' in entry and int(entry['no_good_question']) != 0:
        return False
    
    if 'question_relevant' in entry and int(entry['question_relevant']) != 1:
        return False
    
    if 'resolution_date' not in entry or 'unknown' in entry['resolution_date'].lower():
        return False
    
    # if 'question_start_date' not in entry or 'unknown' in entry['question_start_date'].lower():
    #     return False
    
    
    return True

def extract_news_source_from_filename(filepath: Path) -> str:
    """
    Extract news source from filename pattern like 'deepseek-chat-v3-0324_cnn_7355_free_3.jsonl'
    """
    filename = filepath.name 
    # Remove extension and split by underscore
    name_parts = Path(filename).stem.split('_')
    
    # Common news sources to look for
    news_sources = ['cnn', 'dw', 'forbes', 'reuters', 'cbsnews', 'foxnews', 
                   'theguardian', 'bbc', 'ap', 'npr', 'wsj', 'nyt']
    
    for part in name_parts:
        if part.lower() in news_sources:
            return part.lower()
    
    # Fallback: look for pattern where a part is followed by a number
    for i in range(len(name_parts) - 1):
        if name_parts[i] and name_parts[i+1].isdigit():
            return name_parts[i].lower()
        
    # now search in the whole filepath not just the filename
    print(filepath)
    for part in filepath.parts:
        for news_source in news_sources:
            if news_source in part.lower():
                return part.lower()
    
    return 'unknown'

def convert_single_file(input_file_path: str) -> None:
    """
    Convert a single .jsonl file to standardized format.
    """
    input_path = Path(input_file_path)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_file_path}")
        return
    
    if not input_path.suffix == '.jsonl':
        logger.error(f"Input file must have .jsonl extension: {input_file_path}")
        return
    
    # Create output path with _converted suffix
    output_path = input_path.parent / f"{input_path.stem}_converted.jsonl"
    
    logger.info(f"Converting {input_path.name}")
    logger.info(f"Output file: {output_path}")
    
    # Extract news source from filename
    news_source = extract_news_source_from_filename(input_path)
    logger.info(f"Detected news source: {news_source}")
    
    processed_entries = []
    total_entries = 0
    valid_entries = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as input_file:
            for line_num, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                
                total_entries += 1
                
                try:
                    entry = json.loads(line.strip())
                    
                    if is_valid_news_entry(entry):
                        processed_entries.append(entry)
                        valid_entries += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                
                # Log progress every 10000 entries
                if total_entries % 10000 == 0:
                    logger.info(f"Processed {total_entries} entries, {valid_entries} valid so far")
                    
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    # Write converted entries to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for entry in processed_entries:
                output_file.write(json.dumps(entry) + '\n')
        
        logger.info(f"Successfully converted {len(processed_entries)} entries")
        logger.info(f"Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        return
    
    # Log summary statistics
    logger.info("\n" + "="*50)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*50)
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"News source: {news_source}")
    logger.info(f"Total entries processed: {total_entries}")
    logger.info(f"Valid entries converted: {valid_entries}")
    logger.info(f"Conversion rate: {valid_entries/total_entries*100:.1f}%")
    
    # Show answer type distribution
    answer_types = {}
    for entry in processed_entries:
        answer_type = entry.get('answer_type', 'unknown')
        answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
    
    if answer_types:
        logger.info("Answer type distribution:")
        for answer_type, count in sorted(answer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {answer_type}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a single .jsonl file from news question generation format to standardized format"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to the input .jsonl file to convert'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting single file conversion")
    convert_single_file(args.input_file)
    logger.info("Conversion completed")

if __name__ == "__main__":
    main()
