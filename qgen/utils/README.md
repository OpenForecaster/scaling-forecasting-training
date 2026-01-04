# QGen Utils

Reusable utility modules for data processing, filtering, and conversion.

## Modules

### Core Utilities (Existing)
- **`io_utils.py`** - File I/O (`load_articles_from_file`, `save_jsonl`, `get_files_to_process`)
- **`xml_utils.py`** - XML parsing (`extract_question_components`, `extract_tag_content`)
- **`date_utils.py`** - Date handling (`parse_date`, `normalize_date_format`)

### New Utilities
- **`id_utils.py`** - ID generation (`generate_unique_ids`, `add_ids_to_entries`, `remove_fields_from_entries`)
- **`filtering_utils.py`** - Data validation/filtering (`is_valid_date_format`, `filter_entries`, `has_valid_answer_type`)
- **`conversion_utils.py`** - Format conversion (`extract_news_source_from_filename`, `convert_question_format`, `standardize_entry_format`)

## Usage Examples

### Basic I/O
```python
from qgen.utils import load_articles_from_file, save_jsonl

articles = load_articles_from_file("data.jsonl")
# ... process articles ...
save_jsonl(articles, "output.jsonl")
```

### Adding Unique IDs
```python
from qgen.utils import add_ids_to_entries

entries = add_ids_to_entries(
    entries,
    id_field='question_id',
    seed=42,  # Optional: for reproducibility
    remove_fields=['unwanted_field']
)
```

### Filtering Data
```python
from qgen.utils import filter_entries

result = filter_entries(
    entries,
    cutoff_date="2025-05-01",  # Optional
    explicit_answer_filter=True
)
filtered = result['filtered_entries']
stats = result['stats']
```

### Converting Formats
```python
from qgen.utils import extract_news_source_from_filename, standardize_entry_format
from pathlib import Path

source = extract_news_source_from_filename(Path("www.cnn.com.jsonl"))  # Returns: "cnn"
converted = standardize_entry_format(entry, news_source=source)
```

## CLI Scripts

For command-line usage, see `../combinedata/` directory:
- **add_index.py** - Add random unique IDs
- **clean_articles.py** - Filter and clean articles
- **convert_single_file.py** - Convert question formats
