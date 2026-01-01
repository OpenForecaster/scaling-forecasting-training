#!/usr/bin/env python3
"""
Script to convert Arrow files (downloaded from HuggingFace) to JSONL and Parquet formats.
"""

import os
import json
import pandas as pd
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from typing import Union, Optional

def load_arrow_dataset(dataset_path: str, split: Optional[str] = None):
    """
    Load Arrow dataset from local path or HuggingFace dataset.
    
    Args:
        dataset_path: Path to local Arrow files or HuggingFace dataset name
        split: Dataset split to load (e.g., 'train', 'validation', 'test')
    
    Returns:
        Dataset or DatasetDict object
    """
    print(f"Loading dataset from: {dataset_path}")
    
    if os.path.exists(dataset_path):
        # Local path - load from local Arrow files
        if split:
            dataset = load_dataset('arrow', data_files=f"{dataset_path}/{split}-*.arrow", split=split)
        else:
            dataset = load_dataset('arrow', data_dir=dataset_path)
    else:
        # HuggingFace dataset name
        if split:
            dataset = load_dataset(dataset_path, split=split)
        else:
            dataset = load_dataset(dataset_path)
    
    print(f"Loaded dataset: {type(dataset)}")
    if hasattr(dataset, 'features'):
        print(f"Features: {dataset.features}")
    if hasattr(dataset, '__len__'):
        print(f"Number of examples: {len(dataset)}")
    
    return dataset

def convert_to_dataframe(dataset: Union[Dataset, DatasetDict]) -> pd.DataFrame:
    """
    Convert dataset to pandas DataFrame.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
    
    Returns:
        pandas DataFrame
    """
    if isinstance(dataset, DatasetDict):
        # If it's a DatasetDict, convert the first split to DataFrame
        first_split = list(dataset.keys())[0]
        print(f"Converting split '{first_split}' to DataFrame")
        return dataset[first_split].to_pandas()
    else:
        # If it's a single Dataset
        return dataset.to_pandas()

def save_to_jsonl(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to JSONL format.
    
    Args:
        df: pandas DataFrame
        output_path: Output file path
    """
    print(f"Saving to JSONL: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False, default=str)
            f.write('\n')
    
    print(f"Saved {len(df)} rows to JSONL file")

def save_to_parquet(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: pandas DataFrame
        output_path: Output file path
    """
    print(f"Saving to Parquet: {output_path}")
    
    df.to_parquet(output_path, index=False)
    
    print(f"Saved {len(df)} rows to Parquet file")

def convert_dataset(dataset_path: str, 
                   output_dir: str, 
                   split: Optional[str] = None,
                   formats: list = ['jsonl', 'parquet']):
    """
    Convert Arrow dataset to specified formats.
    
    Args:
        dataset_path: Path to Arrow files or HuggingFace dataset name
        output_dir: Output directory for converted files
        split: Dataset split to convert
        formats: List of output formats ('jsonl', 'parquet')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_arrow_dataset(dataset_path, split)
    
    # Convert to DataFrame
    df = convert_to_dataframe(dataset)
    
    # Generate output filename base
    if isinstance(dataset, DatasetDict):
        split_name = list(dataset.keys())[0] if not split else split
    else:
        split_name = split or 'data'
    
    base_name = f"{split_name}"
    
    # Save in requested formats
    for format_type in formats:
        if format_type.lower() == 'jsonl':
            output_path = os.path.join(output_dir, f"{base_name}.jsonl")
            save_to_jsonl(df, output_path)
        elif format_type.lower() == 'parquet':
            output_path = os.path.join(output_dir, f"{base_name}.parquet")
            save_to_parquet(df, output_path)
        else:
            print(f"Unknown format: {format_type}. Skipping...")

def main():
    parser = argparse.ArgumentParser(description="Convert Arrow files to JSONL/Parquet formats")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to local Arrow files or HuggingFace dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to convert (e.g., 'train', 'validation', 'test')"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs='+',
        default=['jsonl', 'parquet'],
        choices=['jsonl', 'parquet'],
        help="Output formats (default: jsonl parquet)"
    )
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        split=args.split,
        formats=args.formats
    )
    
    print(f"\nConversion completed! Files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
