#!/usr/bin/env python3
"""
Script to combine and split forecasting data from news-generated questions and Metaculus dataset.

This script:
1. Combines .jsonl files from news question generation
2. Processes Metaculus binary data
3. Creates train/validation splits
4. Filters out numeric/date questions for separate splits
"""

import os
import json
import random
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

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
    # Must have final_question as string
    final_question = entry.get('final_question', '')
    if not isinstance(final_question, str) or len(final_question.strip()) < 10:
        return False
    
    # Check validity flags
    if 'final_question_valid' in entry and int(entry['final_question_valid']) != 1:
        return False
    
    if 'no_good_question' in entry and int(entry['no_good_question']) != 0:
        return False
    
    # if 'question_relevant' in entry and int(entry['question_relevant']) != 1:
    #     return False
    
    # Since url will later be used as question id, it must be present
    if 'url' not in entry or len(entry['url']) < 10:
        return False
    
    if 'resolution_date' not in entry or "unknown" in entry['resolution_date'].lower():
        return False
    
    return True

def extract_news_source_from_filename(filename: str) -> str:
    """
    Extract news source from filename pattern like 'deepseek-chat-v3-0324_cnn_7355_free_3.jsonl'
    """
    # Remove extension and split by underscore
    name_parts = Path(filename).stem.split('_')
    
    # Common news sources to look for
    news_sources = ['cnn', 'dw', 'forbes', 'reuters', 'cbsnews', 'foxnews', 'irishtimes', 'hindustantimes'
                   'theguardian', 'bbc', 'ap', 'npr', 'wsj', 'nyt']
    
    for part in name_parts:
        if part.lower() in news_sources:
            return part.lower()
    
    # Fallback: look for pattern where a part is followed by a number
    for i in range(len(name_parts) - 1):
        if name_parts[i] and name_parts[i+1].isdigit():
            return name_parts[i].lower()
    
    return 'unknown'

def process_news_files(input_dir: str) -> tuple[List[Dict[str, Any]], set]:
    """
    Process all .jsonl files in the input directory and extract valid questions.
    Returns (processed_data, news_sources_set)
    """
    processed_data = []
    news_sources = set()
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return [], set()
    
    jsonl_files = list(input_path.glob('*.jsonl'))
    logger.info(f"Found {len(jsonl_files)} .jsonl files")
    
    for file_path in jsonl_files:
        logger.info(f"Processing {file_path.name}")
        
        # Extract news source from filename
        news_source = extract_news_source_from_filename(file_path.name)
        news_sources.add(news_source)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line.strip())
                        
                        if is_valid_news_entry(entry):
                            # Extract question components
                            question_components = extract_question_components(entry['final_question'])
                            
                            # Create standardized entry
                            processed_entry = {
                                'question_title': question_components.get('question_title', ''),
                                'background': question_components.get('background', ''),
                                'resolution_criteria': question_components.get('resolution_criteria', ''),
                                'answer_type': question_components.get('answer_type', ''),
                                'answer': question_components.get('answer', ''),
                                'url': entry.get('url', ''),
                                'article_maintext': entry.get('maintext', ''),
                                'article_publish_date': entry.get('date_publish', ''),
                                'article_modify_date': entry.get('date_modify', ''),
                                'article_download_date': entry.get('date_download', ''),
                                'article_title': entry.get('title', ''),
                                'article_description': entry.get('description', ''),
                                'data_source': f'news_generated_{news_source}',
                                'question_relevant': entry.get('question_relevant', ''),
                                'resolution_date': entry.get('resolution_date', None),
                                'question_start_date': entry.get('question_start_date', None),
                                'news_source': news_source
                            }
                            
                            # Only add if we have a valid question title
                            if processed_entry['question_title'].strip():
                                processed_data.append(processed_entry)
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path.name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    logger.info(f"Processed {len(processed_data)} valid entries from news files")
    logger.info(f"Found news sources: {sorted(news_sources)}")
    return processed_data, news_sources

def process_metaculus_file(metaculus_path: str) -> List[Dict[str, Any]]:
    """
    Process the Metaculus binary dataset file.
    """
    processed_data = []
    
    if not Path(metaculus_path).exists():
        logger.error(f"Metaculus file does not exist: {metaculus_path}")
        return []
    
    logger.info(f"Processing Metaculus file: {metaculus_path}")
    
    try:
        with open(metaculus_path, 'r', encoding='utf-8') as f:
            metaculus_data = json.load(f)
        
        logger.info(f"Loaded {len(metaculus_data)} entries from Metaculus")
        
        for entry in metaculus_data:
            # Determine answer_type based on question_type and resolution
            answer_type = "binary (yes/no)"
            if entry.get('question_type') == 'binary':
                answer_type = "binary"
            
            # Convert resolution to string answer
            resolution = entry.get('resolution')
            if resolution == 1:
                answer = "Yes"
            elif resolution == 0:
                answer = "No"
            else:
                answer = str(resolution) if resolution is not None else ""
            
            processed_entry = {
                'question_title': entry.get('question', ''),
                'background': entry.get('background', ''),
                'resolution_criteria': entry.get('resolution_criteria', ''),
                'answer_type': answer_type,
                'answer': answer,
                'resolution_date': entry.get('date_resolve_at', None),
                'question_close_date': entry.get('date_close', None),
                'question_start_date': entry.get('date_begin', None),
                'url': entry.get('url', ''),
                'article_maintext': '',  # Metaculus doesn't have article text
                'article_publish_date': '',
                'article_modify_date': '',
                'article_download_date': '',
                'data_source': 'metaculus',
                'news_source': 'metaculus',
                'resolution': resolution
            }
            
            if processed_entry['question_title'].strip():
                processed_data.append(processed_entry)
    
    except Exception as e:
        logger.error(f"Error processing Metaculus file: {e}")
        return []
    
    logger.info(f"Processed {len(processed_data)} valid entries from Metaculus")
    return processed_data

def is_numeric_or_date_answer_type(answer_type: str) -> bool:
    """
    Check if answer_type contains numeric or date information.
    """
    answer_type_lower = answer_type.lower()
    numeric_indicators = ['numeric', 'number', 'integer', 'float', 'decimal', 'date', 'time', 'day'] #  'year', 'month', 'day']
    return any(indicator in answer_type_lower for indicator in numeric_indicators)

def save_data_splits(data: List[Dict[str, Any]], output_dir: Path, 
                    train_ratio: float = 1, suffix: str = ""):
    """
    Save data as train/validation splits, and also save all data with suffix 'all'.
    """
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Split data
    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    # Save train split
    train_filename = f"combined{suffix}_train.jsonl"
    train_path = output_dir / train_filename
    with open(train_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    
    # Save validation split
    val_filename = f"combined{suffix}_validation.jsonl"
    val_path = output_dir / val_filename
    with open(val_path, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')
    
    # Save all data
    all_filename = f"combined{suffix}_all.jsonl"
    all_path = output_dir / all_filename
    with open(all_path, 'w', encoding='utf-8') as f:
        for entry in shuffled_data:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"Saved {len(train_data)} train, {len(val_data)} validation, and {len(shuffled_data)} all entries with suffix '{suffix}'")
    return len(train_data), len(val_data)

def main():
    # Configuration
    news_input_dir = "/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/qgen/done"
    metaculus_file = "/fast/nchandak/forecasting/datasets/metaculus/old-2024/binary_raw_train.json"
    # manifold_file = "/fast/nchandak/forecasting/datasets/manifold/manifold_binary_train_with_r1queries.json"
    base_output_dir = "/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix30k-withbinary"
    
    logger.info("Starting data combination and splitting process")
    
    # Process news files
    news_data, news_sources = process_news_files(news_input_dir)
    
    # Process Metaculus file
    metaculus_data = process_metaculus_file(metaculus_file)
    # metaculus_data = []
    # metaculus_data = process_metaculus_file(manifold_file)
    
    # Combine all data
    all_data = news_data + metaculus_data
    logger.info(f"Combined total: {len(all_data)} entries")
    
    if not all_data:
        logger.error("No valid data found. Exiting.")
        return
    
    # Create output directory with news sources in name
    news_sources_sorted = sorted(news_sources - {'unknown'})  # Remove unknown
    if not news_sources_sorted:
        news_sources_sorted = ['mixed']
    subfolder_name = '_'.join(news_sources_sorted)
    output_dir = Path(base_output_dir) / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save full dataset splits
    # train_count, val_count = save_data_splits(all_data, output_dir)
    
    # Filter out numeric/date questions
    non_numeric_data = [
        entry for entry in all_data 
        if not is_numeric_or_date_answer_type(entry.get('answer_type', ''))
    ]
    
    logger.info(f"Filtered to {len(non_numeric_data)} non-numeric/date entries")
    
    # Save non-numeric splits
    if non_numeric_data:
        non_numeric_train_count, non_numeric_val_count = save_data_splits(
            non_numeric_data, output_dir, suffix="_non_numeric"
        )
    
    # Log summary statistics
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"News sources found: {sorted(news_sources)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total entries: {len(all_data)}")
    logger.info(f"  - News generated: {len(news_data)}")
    logger.info(f"  - Metaculus: {len(metaculus_data)}")
    # logger.info(f"Full dataset split:")
    # logger.info(f"  - Train: {train_count}")
    # logger.info(f"  - Validation: {val_count}")
    logger.info(f"Non-numeric dataset: {len(non_numeric_data)} entries")
    if non_numeric_data:
        logger.info(f"  - Train: {non_numeric_train_count}")
        logger.info(f"  - Validation: {non_numeric_val_count}")
    
    # Show some example answer types
    answer_types = {}
    for entry in all_data:
        answer_type = entry.get('answer_type', 'unknown')
        answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
    
    logger.info(f"Answer type distribution:")
    for answer_type, count in sorted(answer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  - {answer_type}: {count}")

if __name__ == "__main__":
    main()
