"""
I/O utility functions for loading and saving data.

This module provides shared functions for:
- Loading articles from JSONL files
- Saving data to JSONL format
- Finding files to process (single file or directory)
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def load_articles_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load articles from a single JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of article dictionaries
        
    Example:
        >>> articles = load_articles_from_file("data/articles.jsonl")
        >>> print(f"Loaded {len(articles)} articles")
    """
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


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
        
    Example:
        >>> articles = [{"title": "Test", "content": "..."}]
        >>> save_jsonl(articles, "output/articles.jsonl")
    """
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(data)} entries to {file_path}")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")


def get_files_to_process(input_path: str, extension: str = '.jsonl') -> List[str]:
    """
    Get list of files to process from a path.
    
    If input_path is a file, returns [input_path].
    If input_path is a directory, returns all files with the given extension.
    
    Args:
        input_path: Path to file or directory
        extension: File extension to filter (default: '.jsonl')
        
    Returns:
        List of file paths to process
        
    Example:
        >>> files = get_files_to_process("data/", ".jsonl")
        >>> print(f"Found {len(files)} files to process")
    """
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix == extension:
            return [str(path)]
        else:
            logger.error(f"Input file must have {extension} extension: {input_path}")
            return []
    elif path.is_dir():
        files = list(path.glob(f'*{extension}'))
        return [str(f) for f in sorted(files)]
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return []

