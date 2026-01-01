# News Article Summarization for Forecasting

This directory contains a modular system for generating and evaluating various types of news article summaries for forecasting questions.

## Overview

The system consists of the following components:

1. **Prompt Templates** (`prompt_templates.py`): Various prompt templates for summarizing news articles with different focuses.
2. **vLLM Summarizer** (`vllm_summarizer.py`): Implementation of the summarization functionality using vLLM with Llama 3.3 70B.
3. **Summary Generator** (`generate_summaries.py`): Main script for generating summaries using different prompts and saving them to datasets.
4. **Summary Analyzer** (`analyze_summaries.py`): Utility for analyzing and comparing generated summaries.

## Prompt Types

The system implements 6 different summarization prompt types:

1. **Basic Summary**: Simple summarization without forecasting context
2. **Forecast-Focused Summary**: Summary that focuses on information relevant to a forecasting question
3. **Key Facts Summary**: Extracts only the key factual information
4. **Forecast Evidence Summary**: Focuses specifically on evidence related to a forecasting question
5. **Timeline-Oriented Summary**: Emphasizes chronology and timeline of events
6. **Halawi**: Prompt template from Halawi et al. (2024)

Each prompt type is tested with 3 different target lengths: 50, 100, and 200 words.

## Usage

### Generating Summaries

To generate summaries with all prompt types and target lengths:

```bash
python generate_summaries.py \
  --source_path "/fast/sgoel/forecasting/news/retrieval/metaculus-binary_reuters_7_365/" \
  --output_dir "/fast/nchandak/forecasting/retrieval_summary/metaculus-binary_reuters_7_365" \
  --model_path "/fast/rolmedo/models/llama-3.3-70b-instruct"
```

Arguments:
- `--source_path`: Path to the source dataset
- `--output_dir`: Directory to save the output datasets
- `--model_path`: Path to the model to use

The system sends all prompts to vLLM at once, which handles batching internally for optimal performance.

### Analyzing Summaries

To analyze the generated summaries:

```bash
python analyze_summaries.py \
  --summaries_dir "/fast/nchandak/forecasting/retrieval_summary/metaculus-binary_reuters_7_365"
```

This will:
1. Calculate statistics for each prompt type and target length
2. Generate plots comparing the different prompt types
3. Sample a few summaries from each prompt type

## Output Datasets

The generated datasets are saved in the specified output directory with filenames in the format:
`{prompt_name}_length{target_length}.dataset`

Each dataset maintains the structure of the original dataset with an additional column:

- `articles_summary`: A list of summary items, one for each article in `retrieved_articles`, with the following fields:
  - `prompt_name`: Name of the prompt used
  - `target_length`: Target length of the summary
  - `model`: Model used for summarization
  - `summary`: The generated summary
  - `news_source`: Source of the news article

## Dependencies

- vLLM
- Hugging Face Datasets
- Pandas
- NumPy
- Matplotlib
- tqdm

## Adding New Prompt Types

To add a new prompt type:

1. Add a new prompt function to `prompt_templates.py`
2. Add the function to the `get_all_prompt_functions()` dictionary

## Adding New Data Sources

To use a different data source:

1. Update the `--source_path` argument to point to the new dataset
2. Update the `--output_dir` argument to reflect the new dataset name 