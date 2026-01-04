"""
Field manipulation utilities for question data.

This module provides utilities for:
- Removing specific fields from entries
- Renaming fields
- Field validation and cleaning
"""

import json
import logging
from typing import List, Dict, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class FieldProcessor:
    """
    Utilities for manipulating fields in question data.
    
    Example:
        >>> processor = FieldProcessor()
        >>> entries = processor.load_from_file("questions.jsonl")
        >>> processor.remove_fields(entries, ['response', 'reasoning'])
        >>> processor.save_to_file(entries, "questions_cleaned.jsonl")
    """
    
    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entries from a JSONL file."""
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        return entries
    
    def save_to_file(self, entries: List[Dict[str, Any]], file_path: str) -> None:
        """Save entries to a JSONL file."""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(entries)} entries to {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
    
    def remove_fields(
        self, 
        entries: List[Dict[str, Any]], 
        fields_to_remove: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Remove specified fields from all entries.
        
        Args:
            entries: List of entry dictionaries
            fields_to_remove: List of field names to remove
            
        Returns:
            List of entries with fields removed (modifies in place)
            
        Example:
            >>> processor = FieldProcessor()
            >>> entries = [{"name": "test", "temp": "delete", "data": "keep"}]
            >>> processor.remove_fields(entries, ["temp"])
            >>> print(entries)  # [{"name": "test", "data": "keep"}]
        """
        fields_set = set(fields_to_remove)
        removed_count = 0
        
        for entry in entries:
            for field in fields_to_remove:
                if field in entry:
                    del entry[field]
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} field instances from {len(entries)} entries")
        return entries
    
    def rename_field(
        self,
        entries: List[Dict[str, Any]],
        old_name: str,
        new_name: str
    ) -> List[Dict[str, Any]]:
        """
        Rename a field in all entries.
        
        Args:
            entries: List of entry dictionaries
            old_name: Current field name
            new_name: New field name
            
        Returns:
            List of entries with field renamed (modifies in place)
        """
        renamed_count = 0
        
        for entry in entries:
            if old_name in entry:
                entry[new_name] = entry.pop(old_name)
                renamed_count += 1
        
        logger.info(f"Renamed field '{old_name}' to '{new_name}' in {renamed_count} entries")
        return entries
    
    def keep_only_fields(
        self,
        entries: List[Dict[str, Any]],
        fields_to_keep: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Keep only specified fields, removing all others.
        
        Args:
            entries: List of entry dictionaries
            fields_to_keep: List of field names to keep
            
        Returns:
            List of entries with only specified fields (modifies in place)
        """
        fields_set = set(fields_to_keep)
        
        for entry in entries:
            keys_to_remove = [key for key in entry.keys() if key not in fields_set]
            for key in keys_to_remove:
                del entry[key]
        
        logger.info(f"Kept only {len(fields_to_keep)} fields in {len(entries)} entries")
        return entries

