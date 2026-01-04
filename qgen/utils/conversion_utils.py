"""
Utilities for converting data formats.

This module provides functions for:
- News source extraction from filenames
- Question format conversion
- Data standardization
"""

import re
from pathlib import Path
from typing import Dict, Any


def extract_news_source_from_filename(filepath: Path) -> str:
    """
    Extract news source from filename.
    
    Handles various patterns:
    - Domain names: www.cnn.com.jsonl -> cnn
    - Pattern-based: deepseek-chat-v3-0324_cnn_7355_free_3.jsonl -> cnn
    - Simple names: guardian_2025.jsonl -> guardian
    
    Args:
        filepath: Path object or string with filename
        
    Returns:
        Extracted news source name (lowercase)
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    filename = filepath.name 
    stem = Path(filename).stem
    
    # First, try to extract from domain names (e.g., www.cnn.com, ca.finance.yahoo.com)
    tlds = ['.com', '.org', '.net', '.edu', '.gov', '.co', '.io', '.ai']
    
    for tld in tlds:
        if tld in stem:
            # Extract the domain part before the TLD
            domain_part = stem.split('_')[0]
            
            # Remove www. prefix if present
            if domain_part.startswith('www.'):
                domain_part = domain_part[4:]
            
            # Split by dots and get the main domain name
            domain_parts = domain_part.split('.')
            if len(domain_parts) >= 2:
                # Get the part before the TLD
                main_domain = domain_parts[-2].lower()
                return main_domain
    
    # If no TLD found, try to match against known news sources
    name_parts = stem.split('_')
    
    # Expanded list of known news sources
    news_sources = [
        'cnn', 'dw', 'forbes', 'reuters', 'cbsnews', 'foxnews', 'time', 'euronews',
        'theguardian', 'guardian', 'bbc', 'ap', 'npr', 'wsj', 'nyt', 'ndtv', 
        'aljazeera', 'independent', 'cnbc', 'bloomberg', 'yahoo', 'arabnews',
        'benzinga', 'seekingalpha', 'fastcompany', 'engadget', 'wired',
        'techcrunch', 'verge', 'ars', 'politico', 'axios', 'vox', 'telegraph'
    ]
    
    # Try to find a known news source in the filename parts
    for part in name_parts:
        if part.lower() in news_sources:
            return part.lower()
    
    # If still not found, look for pattern: word followed by year or number
    for i in range(len(name_parts) - 1):
        if name_parts[i] and name_parts[i+1].isdigit():
            return name_parts[i].lower()
    
    # If no pattern matched, return the first part or full stem
    if '_' not in stem and '.' not in stem:
        return stem.lower()
    
    # Fallback: return stem as-is
    return stem.lower()


def convert_question_format(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert question from XML format to flat dictionary format.
    
    Extracts fields from 'final_question' XML-like structure and 
    flattens them into top-level dictionary fields.
    
    Args:
        entry: Entry dictionary with 'final_question' field
        
    Returns:
        New dictionary with extracted and flattened fields
    """
    from .xml_utils import extract_question_components, extract_tag_content
    
    # Extract question components from final_question
    question_title, background, resolution_criteria = extract_question_components(entry)
    
    # Extract answer and answer_type separately
    final_question = entry.get('final_question', '')
    answer = extract_tag_content(final_question, 'answer') if final_question else entry.get('answer', '')
    answer_type = extract_tag_content(final_question, 'answer_type') if final_question else entry.get('answer_type', '')
    
    # Build new entry with extracted fields
    converted = {
        'question_title': question_title,
        'background': background,
        'resolution_criteria': resolution_criteria,
        'answer': answer,
        'answer_type': answer_type,
    }
    
    # Copy over other fields (metadata)
    metadata_fields = [
        'url', 'article_maintext', 'article_publish_date', 'article_modify_date',
        'article_download_date', 'article_description', 'article_title',
        'data_source', 'news_source', 'original_file', 'resolution_date',
        'resolution_date_response', 'question_start_date', 'question_id'
    ]
    
    for field in metadata_fields:
        if field in entry:
            converted[field] = entry[field]
    
    # Also handle non-prefixed article fields (map them to prefixed versions)
    field_mappings = {
        'title': 'article_title',
        'description': 'article_description',
        'maintext': 'article_maintext',
        'date_publish': 'article_publish_date',
        'date_modify': 'article_modify_date',
        'date_download': 'article_download_date'
    }
    
    for old_field, new_field in field_mappings.items():
        if old_field in entry and new_field not in converted:
            converted[new_field] = entry[old_field]
    
    return converted


def standardize_entry_format(
    entry: Dict[str, Any],
    news_source: str = 'unknown',
    original_filename: str = None
) -> Dict[str, Any]:
    """
    Standardize an entry to a consistent format.
    
    Args:
        entry: Entry dictionary
        news_source: News source name to add
        original_filename: Original filename for reference
        
    Returns:
        Standardized entry dictionary
    """
    # Convert if needed (from XML format)
    if 'final_question' in entry and 'question_title' not in entry:
        entry = convert_question_format(entry)
    
    # Add metadata
    if 'data_source' not in entry:
        entry['data_source'] = 'news_generated'
    
    if 'news_source' not in entry:
        entry['news_source'] = news_source
    
    if original_filename and 'original_file' not in entry:
        entry['original_file'] = original_filename
    
    return entry

