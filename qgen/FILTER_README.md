# Article Filtering with VLLM

This script uses VLLM to evaluate the relevance of news articles for forecasting purposes.

## Features

- Loads articles from a JSONL file
- Uses VLLM with tensor parallel size 8 for efficient processing
- Evaluates each article based on relevance criteria
- Adds a `relevant` field (0 or 1) to each article
- Saves results to a new JSONL file

## Installation

Make sure you have VLLM installed:
```bash
pip install vllm
```

## Usage

### Basic Usage
```bash
python qgen/filter_articles.py \
    --articles_path "/path/to/articles.jsonl" \
    --model_path "/path/to/model"
```

### Full Example
```bash
python qgen/filter_articles.py \
    --articles_path "/fast/sgoel/forecasting/news/tokenized_data/news/deduped/qgen_selected/www.dw.com_2025-05_selected100.jsonl" \
    --model_path "/fast/nchandak/models/Qwen3-32B" \
    --output_path "/path/to/output_filtered.jsonl"
```

## Arguments

- `--articles_path`: Path to the input JSONL file containing articles
- `--model_path`: Path to the VLLM-compatible model directory
- `--output_path`: (Optional) Output path for filtered articles. Defaults to input path with `_filtered` suffix

## Evaluation Criteria

The script evaluates articles based on:
1. **Interest and Reach**: Is the article interesting and relevant to more than 100 people?
2. **Forecasting Value**: If the article covers a specific event, would that event be of interest to people at least one week before it occurred? Would forecasting that event one week before have changed downstream decisions or had other impact?

## Output

The script adds a `relevant` field to each article:
- `"relevant": 1` - Article is relevant for forecasting
- `"relevant": 0` - Article is not relevant for forecasting

## Example Output

```json
{
  "article_title": "Example Article",
  "article_description": "Description...",
  "article_maintext": "Main text...",
  "relevant": 1
}
```

## Performance

- Uses tensor parallel size 8 for efficient GPU utilization
- Processes articles in batches for optimal performance
- VLLM handles internal batching automatically

## Logging

The script provides detailed logging including:
- Number of articles processed
- Evaluation time
- Count of relevant vs non-relevant articles
- Examples of first few evaluations for debugging 