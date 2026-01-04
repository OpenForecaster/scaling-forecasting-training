"""
Utilities for filtering and validating data entries.

This module provides functions for:
- Date validation and filtering
- Answer type validation
- Entry filtering based on various criteria
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional


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


def is_date_after_cutoff(date_str: str, cutoff_date: str) -> bool:
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


def has_valid_answer_type(answer_type: str, explicit_filter: bool = False) -> bool:
    """
    Check if answer_type contains valid keywords.
    
    Args:
        answer_type: Answer type string to check
        explicit_filter: If True, applies strict keyword matching
        
    Returns:
        True if contains valid keywords, False otherwise
    """
    if not answer_type:
        return False
    
    # Convert to lowercase for case-insensitive matching
    answer_type_lower = answer_type.lower()
    
    # Invalid keywords (always filter these out)
    invalid_keywords = [
        "explanation", "any", "integer", "decimal", "percentage", "number", "phrase"
    ]
    
    if any(keyword in answer_type_lower for keyword in invalid_keywords):
        return False
    
    # If not using explicit filter, accept all answer types that don't have invalid keywords
    if not explicit_filter:
        return True
    
    # Valid keywords for explicit filtering
    valid_keywords = [
        "title", "name",
        "month", "year", 
        "location", "place",
        "organization", "organisation", 
        "company", "corporation", "institution", "club", "team",
        "city", "town", "village", 
        "country", "nation", "state", 
        "province", "territory",
        "region", "district", "zone", "sector", "division",
        "county", "department",
        "sport", "game", "league", "competition",  
        "award", "trophy", "medal", "prize", "reward", "honor", "recognition",
        "tournament", "match", "athletics",
        "disease", "syndrome", "disorder", "virus", "bacteria", "event",
        "medical condition", "currency", "brand", 
        "venue", "planet",
    ]
    
    # Check if any keyword exists in answer_type
    return any(keyword in answer_type_lower for keyword in valid_keywords)


def filter_entries(
    entries: List[Dict[str, Any]],
    cutoff_date: Optional[str] = None,
    explicit_answer_filter: bool = False,
    date_field: str = 'resolution_date',
    answer_type_field: str = 'answer_type'
) -> Dict[str, Any]:
    """
    Filter entries based on date and answer type criteria.
    
    Args:
        entries: List of entry dictionaries to filter
        cutoff_date: Optional cutoff date (YYYY-MM-DD). If provided, filters by date
        explicit_answer_filter: If True, applies strict answer type filtering
        date_field: Name of the date field to check (default: 'resolution_date')
        answer_type_field: Name of the answer type field (default: 'answer_type')
        
    Returns:
        Dictionary with:
            - 'filtered_entries': List of entries that passed filters
            - 'stats': Dictionary with filtering statistics
    """
    from .xml_utils import extract_question_components
    
    filtered_entries = []
    stats = {
        'total': len(entries),
        'valid_date': 0,
        'valid_date_cutoff': 0,
        'valid_answer_type': 0,
        'filtered': 0
    }
    
    for entry in entries:
        # Only filter by date if cutoff_date is provided
        if cutoff_date:
            resolution_date = entry.get(date_field, '')
            
            # Skip if no resolution date or invalid format
            if not is_valid_date_format(resolution_date):
                continue
            
            stats['valid_date'] += 1
            
            # Skip if date is not after cutoff
            if not is_date_after_cutoff(resolution_date, cutoff_date):
                continue
            
            stats['valid_date_cutoff'] += 1
        
        # Extract answer_type (might be in nested structure)
        if answer_type_field in entry:
            answer_type = entry[answer_type_field]
        else:
            # Try to extract from question components
            _, _, _, _, answer_type = extract_question_components(entry)
        
        # Skip if no valid answer_type
        if not has_valid_answer_type(answer_type, explicit_answer_filter):
            continue
        
        stats['valid_answer_type'] += 1
        
        # Entry passed all filters
        filtered_entries.append(entry)
    
    stats['filtered'] = len(filtered_entries)
    
    return {
        'filtered_entries': filtered_entries,
        'stats': stats
    }


def is_valid_news_entry(entry: Dict[str, Any]) -> bool:
    """
    Check if a news entry meets basic validity criteria.
    
    Args:
        entry: Entry dictionary to validate
        
    Returns:
        True if entry is valid, False otherwise
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
    
    if 'question_relevant' in entry and int(entry['question_relevant']) != 1:
        return False
    
    # Check resolution_date if present
    if 'resolution_date' in entry:
        if entry['resolution_date'] == '':
            return False
        
        resolution_date = entry['resolution_date']
        try:
            datetime.strptime(resolution_date, '%Y-%m-%d')
        except Exception:
            return False
    
    # Check question_start_date if present
    if 'question_start_date' in entry and entry['question_start_date'] == '':
        return False
    
    return True

