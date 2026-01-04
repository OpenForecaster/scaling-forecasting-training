#!/usr/bin/env python3
"""
Script to upload train and validation files to HuggingFace dataset repository.
"""

import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import argparse
from pathlib import Path

def load_parquet_file(file_path):
    """Load a Parquet file and return a pandas DataFrame."""
    print(f"Loading Parquet file: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Loaded DataFrame with shape: {df.shape}")
    return df

def create_dataset_from_parquet(train_file, val_file):
    """Create a DatasetDict from train and validation Parquet files."""
    print(f"Loading train file: {train_file}")
    train_df = load_parquet_file(train_file)
    print(f"Loaded {len(train_df)} training examples")
    
    print(f"Loading validation file: {val_file}")
    val_df = load_parquet_file(val_file)
    print(f"Loaded {len(val_df)} validation examples")
    
    # Create datasets from DataFrames
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    return dataset_dict

def upload_to_hub(dataset_dict, repo_id, token=None):
    """Upload dataset to HuggingFace Hub."""
    print(f"Uploading dataset to: {repo_id}")
    
    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=False,  # Set to True if you want a private dataset
        commit_message="Upload freeform forecasting dataset"
    )
    
    print(f"Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload forecasting dataset to HuggingFace")
    parser.add_argument(
        "--train-file", 
        type=str, 
        default="/fast/nchandak/forecasting/datasets/verl/freeform/data70k-withbinary2k/combined_non_numeric_all_train.jsonl",
        help="Path to training Parquet file"
    )
    parser.add_argument(
        "--val-file", 
        type=str, 
        default="/fast/nchandak/forecasting/datasets/verl/freeform/data70k-withbinary2k/combined_non_numeric_all_validation.jsonl",
        help="Path to validation Parquet file"
    )
    parser.add_argument(
        "--repo-id", 
        type=str, 
        default="nikhilchandak/freeform-forecasting",
        help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="HuggingFace token (if not provided, will use environment variable HF_TOKEN)"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not os.path.exists(args.val_file):
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")
    
    # Get token from environment if not provided
    token = args.token or os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("HuggingFace token is required. Set HF_TOKEN environment variable or use --token argument.")
    
    # Create dataset
    dataset_dict = create_dataset_from_parquet(args.train_file, args.val_file)
    
    # Print dataset info
    print("\nDataset Info:")
    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")
    print(f"Features: {dataset_dict['train'].features}")
    
    # Upload to hub
    upload_to_hub(dataset_dict, args.repo_id, token)
    
    print("Upload completed successfully!")

if __name__ == "__main__":
    main()
