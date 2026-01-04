"""
Date parsing and formatting utilities.

This module provides functions for:
- Parsing dates from various string formats
- Normalizing date formats
- Converting between different date representations
"""

import re
from datetime import datetime
from typing import Optional


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string and return a datetime object.
    
    Handles multiple date formats including:
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM:SS
    - ISO 8601 formats
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object with timezone info removed, or None if parsing fails
        
    Example:
        >>> dt = parse_date("2024-03-15")
        >>> print(dt.year, dt.month, dt.day)
        2024 3 15
    """
    if not date_str or date_str is None:
        return None
    
    date_str = str(date_str).strip()
    
    # Handle different date formats
    date_formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S%z',
        '%Y-%m-%d %H:%M:%S+00:00',
        '%Y-%m-%dT%H:%M:%SZ',  # ISO 8601 with Z suffix
        '%Y-%m-%dT%H:%M:%S',   # ISO 8601 without timezone
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Convert to naive datetime (remove timezone info for comparison)
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    
    # Try to handle various timezone formats using fromisoformat
    try:
        # Handle Z suffix (Zulu time)
        if date_str.endswith('Z'):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        elif '+00:00' in date_str:
            dt = datetime.fromisoformat(date_str.replace('+00:00', '+0000'))
        else:
            dt = datetime.fromisoformat(date_str)
        # Convert to naive datetime (remove timezone info for comparison)
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


def format_month_day_year(month_str: str, day_str: str, year_str: str) -> str:
    """
    Convert 'Month DD, YYYY' to 'YYYY-MM-DD' format.
    
    Args:
        month_str: Month name (e.g., "March", "mar")
        day_str: Day as string (e.g., "15")
        year_str: Year as string (e.g., "2024")
        
    Returns:
        Formatted date string 'YYYY-MM-DD', or empty string if month not recognized
        
    Example:
        >>> format_month_day_year("March", "15", "2024")
        '2024-03-15'
    """
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
    """
    Convert 'DD Month YYYY' to 'YYYY-MM-DD' format.
    
    Args:
        day_str: Day as string (e.g., "15")
        month_str: Month name (e.g., "March", "mar")
        year_str: Year as string (e.g., "2024")
        
    Returns:
        Formatted date string 'YYYY-MM-DD', or empty string if month not recognized
        
    Example:
        >>> format_day_month_year("15", "March", "2024")
        '2024-03-15'
    """
    return format_month_day_year(month_str, day_str, year_str)


def normalize_date_format(date_text: str) -> str:
    """
    Normalize various date formats to a consistent YYYY-MM-DD format.
    
    Handles formats like:
    - "March 15, 2024"
    - "15 March 2024"
    - "15th March 2024"
    - "2024-03-15"
    - "March 2024" (returns YYYY-MM format)
    
    Args:
        date_text: Date string in various formats
        
    Returns:
        Normalized date string, or original text if can't parse
        
    Example:
        >>> normalize_date_format("March 15, 2024")
        '2024-03-15'
        >>> normalize_date_format("15th March 2024")
        '2024-03-15'
    """
    if not date_text:
        return ""
    
    # Remove common prefixes/suffixes
    date_text = re.sub(r'^\s*(on\s+|the\s+)', '', date_text, flags=re.IGNORECASE)
    date_text = re.sub(r'\s*(onwards?|forward)\s*$', '', date_text, flags=re.IGNORECASE)
    
    # Common date patterns to recognize and normalize
    date_patterns = [
        # YYYY-MM-DD format (already normalized)
        (r'^(\d{4})-(\d{1,2})-(\d{1,2})$', r'\1-\2-\3'),
        # DD/MM/YYYY or MM/DD/YYYY
        (r'^(\d{1,2})/(\d{1,2})/(\d{4})$', r'\3-\2-\1'),
        # DD.MM.YYYY
        (r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', r'\3-\2-\1'),
    ]
    
    for pattern, replacement in date_patterns:
        if re.search(pattern, date_text):
            return re.sub(pattern, replacement, date_text)
    
    # Month DD, YYYY (e.g., "March 10, 2024")
    match = re.search(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_text, re.IGNORECASE)
    if match:
        result = format_month_day_year(match.group(1), match.group(2), match.group(3))
        if result:
            return result
    
    # DD Month YYYY (e.g., "10 March 2024" or "10th March 2024")
    match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})', date_text, re.IGNORECASE)
    if match:
        result = format_day_month_year(match.group(1), match.group(2), match.group(3))
        if result:
            return result
    
    # Month YYYY (e.g., "March 2024")
    match = re.search(r'^(\w+)\s+(\d{4})$', date_text, re.IGNORECASE)
    if match:
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
            'oct': '10', 'nov': '11', 'dec': '12'
        }
        month_num = month_map.get(match.group(1).lower())
        if month_num:
            return f"{match.group(2)}-{month_num}"
    
    # If no pattern matches, return the original text (might be a valid date description)
    return date_text.strip()

