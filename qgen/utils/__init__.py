"""
Shared utility functions for the qgen module.

This package contains reusable utilities for:
- I/O operations (JSONL reading/writing)
- XML parsing and extraction
- Date parsing and formatting
- ID generation and management
- Data filtering and validation
- Format conversion
"""

# I/O utilities
from .io_utils import (
    load_articles_from_file,
    save_jsonl,
    get_files_to_process
)

# XML utilities
from .xml_utils import (
    extract_question_components,
    extract_tag_content,
    extract_final_question
)

# Date utilities
from .date_utils import (
    parse_date,
    normalize_date_format,
    format_month_day_year,
    format_day_month_year
)

# ID utilities
from .id_utils import (
    generate_unique_ids,
    add_ids_to_entries,
    remove_fields_from_entries
)

# Filtering utilities
from .filtering_utils import (
    is_valid_date_format,
    is_date_on_or_after_first_date,
    has_valid_answer_type,
    filter_entries,
    is_valid_news_entry
)

# Conversion utilities
from .conversion_utils import (
    extract_news_source_from_filename,
    convert_question_format,
    standardize_entry_format
)

__all__ = [
    # I/O utilities
    'load_articles_from_file',
    'save_jsonl',
    'get_files_to_process',
    
    # XML utilities
    'extract_question_components',
    'extract_tag_content',
    'extract_final_question',
    
    # Date utilities
    'parse_date',
    'normalize_date_format',
    'format_month_day_year',
    'format_day_month_year',
    
    # ID utilities
    'generate_unique_ids',
    'add_ids_to_entries',
    'remove_fields_from_entries',
    
    # Filtering utilities
    'is_valid_date_format',
    'is_date_on_or_after_first_date',
    'has_valid_answer_type',
    'filter_entries',
    'is_valid_news_entry',
    
    # Conversion utilities
    'extract_news_source_from_filename',
    'convert_question_format',
    'standardize_entry_format',
]

