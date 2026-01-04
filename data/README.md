# Data Directory

This directory contains scripts and data for processing forecasting questions from various platforms.

## Directory Structure

### Root Level Scripts
- **`manifold_new.py`** - Processes Manifold Markets data dumps to standardized format
- **`futureX.py`** - Downloads and processes FutureX-Online dataset from HuggingFace
- **`process_relevant_docs.py`** - Combines and ranks retrieved documents for RAG
- **`filter_metaculus_nulls.py`** - Filters out Metaculus questions with null resolutions

### Subdirectories

#### `metaculus/`
Scripts for fetching and processing Metaculus forecasting questions:
- **`metaculus_new.py`** - Main API v2 fetcher with date filtering
- **`get_metaculus.py`** - Enhanced fetcher with custom date ranges
- **`metaculus.py`** - Legacy scraper (from consistency-forecasting repo)
- **`prepare_metaculus_test.py`** - Creates test datasets with prompts
- **`prepare_metaculus_train.py`** - Creates training datasets
- **`convert_to_standardized_format.py`** - Converts to standard format
- **`decide_dates_real_fq.py`** - Date filtering logic for questions
- **`filter_all.py`** - Quality filtering and deduplication
- **`data/`** - Intermediate data files (.json, .jsonl)

#### `new_manifold/`
Manifold Markets data processing:
- **`manifold.py`** - Scrapes and processes Manifold API data
- **`create_test_set.py`** - Creates filtered test sets with quality checks
- **`manifold_split_data.py`** - Splits data into train/test
- **`mcq_analyze_manifold.py`** - Analyzes MCQ questions

#### `foresight/`
OpenForesight dataset preparation:
- **`standardize_data.py`** - Standardizes question format and generates prompts
- **`push_to_hf.py`** - Uploads dataset to HuggingFace Hub
- **`push_model.py`** - Uploads trained models to HuggingFace Hub

#### `theguardian/`
The Guardian news article fetching:
- **`guardian_fetcher_3months.py`** - Fetches articles for 3-month ranges
- **`guardian_fetcher_july.py`** - Fetches July 2025 articles
- **`deduplicate_articles.py`** - Removes duplicate articles
- **`sample_articles.py`** - Samples articles for testing
- **`apikey.py`** - API key configuration

#### `retrieval_summary/`
News article summarization for RAG:
- **`generate_summaries.py`** - Generates summaries with multiple prompt types
- **`vllm_summarizer.py`** - vLLM-based summarization
- **`prompt_templates.py`** - Different summarization prompts
- **`analyze_summaries.py`** - Analyzes generated summaries
- **`analysis_reuters.py`** - Reuters-specific analysis

#### `kalshi/`
- **`fetcher.py`** - Fetches resolved MCQ events from Kalshi API

#### `hf_futurebench/`
- **`check.py`** - Analyzes FutureBench dataset from HuggingFace

#### `infinitegames/`
- **`analyze_binary.py`** - Analyzes Infinite Games binary questions
- **`analysis.ipynb`** - Jupyter notebook for exploration

## Data Sources

- **Metaculus**: Professional forecasting platform (https://www.metaculus.com)
- **Manifold Markets**: Prediction markets (https://manifold.markets, data: https://docs.manifold.markets/api)
- **FutureX**: FutureX-Online benchmark dataset
- **The Guardian**: News articles via Open Platform API
- **Kalshi**: Event-based prediction markets
- **FutureBench**: Standardized forecasting benchmark
- **Infinite Games**: LLM-based forecasting platform 