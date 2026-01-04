"""
Utilities for generating and managing unique identifiers.

This module provides functions for:
- Generating random unique IDs
- Adding IDs to data entries
- Removing unwanted fields
"""

import random
from typing import List, Dict, Any, Set, Optional


def generate_unique_ids(
    count: int, 
    existing_ids: Set[int] = None, 
    start_idx: int = 0,
    max_id: int = None,
    seed: Optional[int] = None
) -> List[int]:
    """
    Generate a list of random unique IDs that don't collide with existing ones.
    
    Args:
        count: Number of IDs to generate
        existing_ids: Set of already existing IDs to avoid (default: empty set)
        start_idx: Minimum value for IDs (default: 0)
        max_id: Maximum value for IDs (default: max(start_idx + count * 10, 1000000))
        seed: Random seed for reproducibility (optional)
        
    Returns:
        List of random unique integer IDs
        
    Raises:
        ValueError: If not enough available IDs in the range
    """
    if seed is not None:
        random.seed(seed)
    
    if existing_ids is None:
        existing_ids = set()
    
    if max_id is None:
        max_id = max(start_idx + count * 10, 1000000)
    
    # Generate pool of available IDs
    available_ids = set(range(start_idx, max_id)) - existing_ids
    
    if len(available_ids) < count:
        raise ValueError(
            f"Not enough available IDs. Need {count}, have {len(available_ids)}"
        )
    
    # Randomly sample the needed IDs
    return random.sample(list(available_ids), count)


def add_ids_to_entries(
    entries: List[Dict[str, Any]],
    id_field: str = 'question_id',
    start_idx: int = 0,
    seed: Optional[int] = None,
    remove_fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Add random unique IDs to entries that don't have them.
    
    Args:
        entries: List of dictionary entries
        id_field: Name of the ID field (default: 'question_id')
        start_idx: Minimum value for random IDs (default: 0)
        seed: Random seed for reproducibility (optional)
        remove_fields: List of field names to remove from all entries (optional)
        
    Returns:
        List of entries with IDs added (modifies entries in-place)
    """
    if seed is not None:
        random.seed(seed)
    
    # Collect existing IDs and entries needing IDs
    existing_ids = set()
    entries_needing_id = []
    
    for idx, entry in enumerate(entries):
        # Remove unwanted fields if specified
        if remove_fields:
            for field in remove_fields:
                entry.pop(field, None)
        
        # Check if ID already exists
        if id_field in entry:
            existing_ids.add(entry[id_field])
        else:
            entries_needing_id.append(idx)
    
    # Generate new IDs for entries that need them
    if entries_needing_id:
        new_ids = generate_unique_ids(
            count=len(entries_needing_id),
            existing_ids=existing_ids,
            start_idx=start_idx,
            seed=seed
        )
        
        # Assign the new IDs
        for entry_idx, new_id in zip(entries_needing_id, new_ids):
            entries[entry_idx][id_field] = new_id
    
    return entries


def remove_fields_from_entries(
    entries: List[Dict[str, Any]],
    fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Remove specified fields from all entries.
    
    Args:
        entries: List of dictionary entries
        fields: List of field names to remove
        
    Returns:
        List of entries with fields removed (modifies entries in-place)
    """
    for entry in entries:
        for field in fields:
            entry.pop(field, None)
    
    return entries

