# Combinedata Scripts

CLI scripts for processing and combining question data.

## Scripts

### 1. `convert_single_file.py`
Convert JSONL from question generation format to standardized format.

```bash
# Basic (auto-detect news source)
python convert_single_file.py --input_file articles.jsonl

# With options
python convert_single_file.py --input_file articles.jsonl \
    --news_source cnn --skip_invalid --output_file converted.jsonl
```

### 2. `clean_articles.py`
Filter articles by date and answer type.

```bash
# Answer type only (lenient)
python clean_articles.py --input_path data/

# With strict filtering
python clean_articles.py --input_path data/ --explicit_filter

# With date filter
python clean_articles.py --input_path data/ --cutoff_date 2025-05-01

# Combined
python clean_articles.py --input_path data/ \
    --cutoff_date 2025-05-01 --explicit_filter
```

**Output:** Creates `cleaned/` subdirectory with `*_cleaned.jsonl` files.

### 3. `add_index.py`
Add random unique `question_id` to entries.

```bash
# In-place modification
python add_index.py --input_file questions.jsonl

# Save to new file
python add_index.py --input_file questions.jsonl --output_file indexed.jsonl

# With reproducible seed
python add_index.py --input_file questions.jsonl --seed 42

# Dry run
python add_index.py --input_file questions.jsonl --dry-run
```

## Typical Workflow

```bash
# Step 1: Convert
python convert_single_file.py --input_file raw.jsonl --skip_invalid

# Step 2: Clean
python clean_articles.py --input_path raw_converted.jsonl --cutoff_date 2025-05-01

# Step 3: Add IDs
python add_index.py --input_file cleaned/raw_converted_cleaned.jsonl --seed 42
```

## Batch Processing

```bash
# Convert all files in directory
for file in raw/*.jsonl; do
    python convert_single_file.py --input_file "$file" --skip_invalid
done

# Clean all
python clean_articles.py --input_path raw/ --cutoff_date 2025-05-01

# Add IDs to all cleaned files
for file in raw/cleaned/*.jsonl; do
    python add_index.py --input_file "$file" --seed 42
done
```

## Python API

Use utilities directly for more control:

```python
from qgen.utils import (
    load_articles_from_file, save_jsonl,
    add_ids_to_entries, filter_entries, standardize_entry_format
)

entries = load_articles_from_file("data.jsonl")
entries = [standardize_entry_format(e, "cnn") for e in entries]
result = filter_entries(entries, cutoff_date="2025-05-01")
entries = add_ids_to_entries(result['filtered_entries'], seed=42)
save_jsonl(entries, "output.jsonl")
```

See `../utils/README.md` for utility documentation.
