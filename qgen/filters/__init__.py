"""
Filtering operations for articles and questions.

This package contains:
- Article filtering by relevance
- Leakage detection and removal
"""

from .leakage_filter import (
    LeakageRemover, 
    filter_by_leakage, 
    remove_exact_leakage_patterns,
    remove_exact_leakage_from_entries
)

__all__ = [
    'LeakageRemover',
    'filter_by_leakage',
    'remove_exact_leakage_patterns',
    'remove_exact_leakage_from_entries',
]

