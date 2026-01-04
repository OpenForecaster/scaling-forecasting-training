#!/usr/bin/env python3
"""
Script to combine and split forecasting data from news-generated questions and Metaculus dataset.

This script:
1. Loads q1, q2, q3 questions from specified .jsonl files
2. Processes Metaculus binary data
3. Creates train/validation splits
4. Filters out numeric/date questions for separate splits
"""

import os
import json
import random
import re
import argparse
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

def extract_question_from_xml(xml_text: str) -> Dict[str, str]:
    """
    Extract question components from XML-formatted question (q1, q2, q3 format).
    Returns dict with question_title, background, resolution_criteria, answer, answer_type
    """
    def extract_tag_content(text: str, tag: str) -> str:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        start = text.find(open_tag)
        if start == -1:
            return ""
        start += len(open_tag)
        end = text.find(close_tag, start)
        if end == -1:
            return ""
        return text[start:end].strip()

    tags = ["question_title", "background", "resolution_criteria", "answer", "answer_type"]
    result = {tag: extract_tag_content(xml_text, tag) for tag in tags}
    return result

def process_q1q2q3_files(input_file: str) -> List[Dict[str, Any]]:
    """
    Process .jsonl file containing q1, q2, q3 questions and extract all questions.
    Returns list of processed question entries.
    """
    processed_data = []
    
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return []
    
    logger.info(f"Processing file: {input_file}")
    
    # Extract news source from filename
    news_source = extract_news_source_from_filename(input_path.name)
    leakage = 0 
    total = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                if line_num > 20000:
                    break 
                
                try:
                    entry = json.loads(line.strip())
                    
                    # Extract questions from q1, q2, q3 fields
                    questions = []
                    for q_field in ['q1', 'q2', 'q3']:
                        if q_field in entry and entry[q_field]:
                            q_xml = entry[q_field]
                            q_components = extract_question_from_xml(q_xml)
                            q_answer = q_components.get('answer', 'N/A')
                            
                            # Only add if we have a valid question title
                            if q_components.get('question_title', '').strip():
                                processed_entry = {
                                    # 'question_title': q_components.get('question_title', ''),
                                    # 'background': q_components.get('background', ''),
                                    # 'resolution_criteria': q_components.get('resolution_criteria', ''),
                                    # 'answer_type': q_components.get('answer_type', ''),
                                    # 'answer': q_components.get('answer', ''),
                                    'final_question': q_xml,
                                    'url': entry.get('url', ''),
                                    'article_maintext': entry.get('maintext', ''),
                                    'article_publish_date': entry.get('date_publish', ''),
                                    'article_modify_date': entry.get('date_modify', ''),
                                    'article_download_date': entry.get('date_download', ''),
                                    'article_title': entry.get('title', ''),
                                    'article_description': entry.get('description', ''),
                                    'data_source': f'news_generated_unfiltered_{news_source}_{q_field}',
                                    'question_relevant': entry.get('question_relevant', ''),
                                    'resolution_date': entry.get('resolution_date', None),
                                    'question_start_date': entry.get('question_start_date', None),
                                    'news_source': news_source,
                                    'question_field': q_field  # Track which field this came from
                                }
                                
                                if q_answer in q_components['question_title'] or q_answer in q_components['background'] or q_answer in q_components['resolution_criteria']:
                                    leakage += 1
                                # if q_answer in processed_entry['final_question']:
                                #     leakage += 1
                                total += 1
                                
                                questions.append(processed_entry)
                    
                    processed_data.extend(questions)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return []
    
    logger.info(f"Processed {len(processed_data)} questions from q1, q2, q3 fields")
    logger.info(f"Leakage: {leakage} / {total}")
    return processed_data

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
    
    # if 'resolution_date' not in entry or entry['resolution_date'] is None:
    #     return False
    
    # if 'question_start_date' not in entry or entry['question_start_date'] is None:
    #     return False
    
    # # remove entries before 2020-01-01
    # if entry['question_start_date'] < '2020-01-01':
    #     return False
    
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

def process_news_files(input_file: str) -> List[Dict[str, Any]]:
    """
    Process all .jsonl files in the input directory and extract valid questions.
    Returns processed_data
    """
    processed_data = []
    
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return []
    
    logger.info(f"Processing file: {input_file}")
    
    # Extract news source from filename
    news_source = extract_news_source_from_filename(input_path.name)
    leakage = 0 
    total = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                if line_num > 60000:
                    break 
                
                try:
                    entry = json.loads(line.strip())
                    
                    if is_valid_news_entry(entry):
                        # Extract question components
                        question_components = extract_question_components(entry['final_question'])
                        q_answer = question_components.get('answer', 'N/A')
                        
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
                        
                        if q_answer in processed_entry['question_title'] or q_answer in processed_entry['background'] or q_answer in processed_entry['resolution_criteria']:
                            leakage += 1
                        total += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return []
    
    logger.info(f"Processed {len(processed_data)} valid entries from news files")
    logger.info(f"Leakage: {leakage} / {total}")
    return processed_data

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine and process forecasting data')
    parser.add_argument('--input-file', type=str, 
                       default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/qgen/clean/deepseek-chat-v3-0324_forbes-2024_60178_free_3.jsonl",
                       help='Input .jsonl file containing q1, q2, q3 questions')
    parser.add_argument('--output-dir', type=str,
                       default="/fast/nchandak/forecasting/datasets/synthetic/freeform/forbes23/",
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    logger.info("Starting data combination and splitting process")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process q1, q2, q3 file
    # q1q2q3_data = process_q1q2q3_files(args.input_file)
    q1q2q3_data = process_news_files(args.input_file)
    
    
    # Filter out numeric/date questions
    non_numeric_data = [
        entry for entry in q1q2q3_data 
        if not is_numeric_or_date_answer_type(entry.get('answer_type', ''))
    ]
    non_numeric_data = non_numeric_data[:10000]
    
    # Combine all data
    all_data = non_numeric_data 
    logger.info(f"Combined total: {len(all_data)} entries")
    
    if not all_data:
        logger.error("No valid data found. Exiting.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save all data as train split (no filtering, no validation split)
    # train_filename = "unfiltered_train30k.jsonl"
    train_filename = "filtered_train_10k.jsonl"
    train_path = output_dir / train_filename
    with open(train_path, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"Saved {len(all_data)} entries as train split")
    
    # Log summary statistics
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total entries: {len(all_data)}")
    logger.info(f"  - Q1Q2Q3 questions: {len(q1q2q3_data)}")
    logger.info(f"  - Non-numeric/date questions: {len(non_numeric_data)}")
    logger.info(f"Train split: {len(all_data)} entries")
    
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
