#!/usr/bin/env python3
"""
Script to process Manifold binary dataset and extract the last 2000 questions sorted by resolution date.

This script:
1. Loads the Manifold binary raw train data
2. Sorts by date_resolve_at
3. Takes the last 2000 questions
4. Saves them in JSONL format
"""

import os
import json
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

def process_manifold_file(manifold_path: str) -> List[Dict[str, Any]]:
    """
    Process the Manifold binary dataset file.
    """
    processed_data = []
    
    if not Path(manifold_path).exists():
        logger.error(f"Manifold file does not exist: {manifold_path}")
        return []
    
    logger.info(f"Processing Manifold file: {manifold_path}")
    
    try:
        with open(manifold_path, 'r', encoding='utf-8') as f:
            manifold_data = json.load(f)
        
        logger.info(f"Loaded {len(manifold_data)} entries from Manifold")
        
        for entry in manifold_data:
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
                'article_maintext': '',  # Manifold doesn't have article text
                'article_publish_date': '',
                'article_modify_date': '',
                'article_download_date': '',
                'data_source': 'manifold',
                'news_source': 'manifold',
                'resolution': resolution
            }
            
            if processed_entry['question_title'].strip():
                processed_data.append(processed_entry)
    
    except Exception as e:
        logger.error(f"Error processing Manifold file: {e}")
        return []
    
    logger.info(f"Processed {len(processed_data)} valid entries from Manifold")
    return processed_data

def save_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save data to JSONL format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} entries to {output_path}")

def main():
    # Input and output paths
    input_path = "/fast/nchandak/forecasting/datasets/manifold/binary_raw_train.json"
    output_dir = "/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval"
    output_filename = "binary_train_2k.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    logger.info("Starting Manifold binary data processing...")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    
    # Process the Manifold file
    processed_data = process_manifold_file(input_path)
    
    if not processed_data:
        logger.error("No data processed from Manifold file")
        return
    
    # Filter out entries without resolution_date
    filtered_data = [entry for entry in processed_data if entry.get('resolution_date') is not None]
    logger.info(f"Filtered to {len(filtered_data)} entries with resolution dates")
    
    if len(filtered_data) == 0:
        logger.error("No entries with resolution dates found")
        return
    
    # Sort by resolution date (ascending order)
    try:
        sorted_data = sorted(filtered_data, key=lambda x: x['resolution_date'])
        logger.info(f"Sorted {len(sorted_data)} entries by resolution date")
    except Exception as e:
        logger.error(f"Error sorting data by resolution date: {e}")
        return
    
    # Take the last 2000 questions
    last_2k = sorted_data[-2000:] if len(sorted_data) >= 2000 else sorted_data
    logger.info(f"Selected the last {len(last_2k)} questions")
    
    if len(last_2k) < 2000:
        logger.warning(f"Only {len(last_2k)} questions available, less than requested 2000")
    
    # Log some statistics about the selected data
    if last_2k:
        earliest_date = last_2k[0]['resolution_date']
        latest_date = last_2k[-1]['resolution_date']
        logger.info(f"Date range: {earliest_date} to {latest_date}")
        
        # Count by resolution
        yes_count = sum(1 for entry in last_2k if entry['answer'] == 'Yes')
        no_count = sum(1 for entry in last_2k if entry['answer'] == 'No')
        other_count = len(last_2k) - yes_count - no_count
        logger.info(f"Resolution breakdown: Yes={yes_count}, No={no_count}, Other={other_count}")
    
    # Save to JSONL
    save_jsonl(last_2k, output_path)
    
    logger.info("Processing completed successfully!")

if __name__ == "__main__":
    main()
