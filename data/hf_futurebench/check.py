#!/usr/bin/env python3
"""
FutureBench Dataset Analyzer

Purpose:
    Loads and analyzes the FutureBench dataset from HuggingFace.
    Filters to 'news' event type and explores data structure.

Main Operations:
    - Load dataset from futurebench/data
    - Filter to news event_type
    - Show statistics and sample data
    - Analyze event_ids, date ranges, and columns

Output:
    Console output with dataset statistics and sample rows

Usage:
    python check.py
"""

from datasets import load_dataset
import pandas as pd

# Load the dataset from HuggingFace
print("Loading dataset from HuggingFace...")
dataset = load_dataset("futurebench/data")

# Convert to pandas DataFrame for easier manipulation
df = dataset['train'].to_pandas()

# Print first few rows and columns to understand the data
print("\n=== First few rows and columns ===")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Print data types and basic info
print("\n=== Dataset info ===")
print(df.info())

# Filter to event_type 'news' and only rows with unique event_id
print("\n=== Filtering data ===")
print(f"Original dataset size: {len(df)}")

# Check unique event_types
print(f"Unique event_types: {df['event_type'].unique()}")

# Filter to event_type 'news'
news_df = df[df['event_type'] == 'news']
print(f"After filtering to 'news' event_type: {len(news_df)}")

# Filter to only rows with unique event_id
# unique_news_df = news_df.drop_duplicates(subset=['event_id'])
# print(f"After filtering to unique event_id: {len(unique_news_df)}")
unique_news_df = news_df

# Print some rows from the remaining data
print("\n=== Sample rows from filtered dataset ===")
print(unique_news_df.head(10))

# Print size of the final dataset
print(f"\n=== Final dataset size ===")
print(f"Number of rows: {len(unique_news_df)}")
print(f"Number of columns: {len(unique_news_df.columns)}")

# Show some statistics about the filtered data
print("\n=== Statistics about filtered data ===")
print(f"Unique event_ids: {unique_news_df['event_id'].nunique()}")
print(f"Date range (open_to_bet_until): {unique_news_df['open_to_bet_until'].min()} to {unique_news_df['open_to_bet_until'].max()}")
print(f"Prediction date range: {unique_news_df['prediction_created_at'].min()} to {unique_news_df['prediction_created_at'].max()}")
