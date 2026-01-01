#!/usr/bin/env python3
"""
Script to upload the OpenForesight dataset to Hugging Face.

Before running this script, ensure:
1. Activate the forecast uv environment: source <path_to_forecast_env>/bin/activate
2. Load CUDA module: module load cuda/12.1

Or run with:
    module load cuda/12.1 && source <forecast_env>/bin/activate && python push_to_hf.py
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi
import os

def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def create_dataset_dict(data_dir):
    """Create a DatasetDict from the standardized files with explicit features"""
    
    # Define the features schema explicitly based on the actual data structure
    features = Features({
        'qid': Value('string'),
        'question_title': Value('string'),
        'background': Value('string'),
        'resolution_criteria': Value('string'),
        'answer_type': Value('string'),
        'answer': Value('string'),
        'url': Value('string'),
        'article_maintext': Value('string'),
        'article_publish_date': Value('string'),
        'article_modify_date': Value('string'),
        'article_download_date': Value('string'),
        'article_description': Value('string'),
        'article_title': Value('string'),
        'data_source': Value('string'),
        'news_source': Value('string'),
        'resolution_date': Value('string'),
        'question_start_date': Value('string'),
        'prompt': Value('string'),
        'prompt_without_retrieval': Value('string'),
    })
    
    splits = {}
    
    # Process train, validation, and test splits
    for split_name in ['train', 'validation', 'test']:
        file_path = data_dir / f"{split_name}.jsonl"
        if file_path.exists():
            print(f"Loading {split_name} split...")
            data = load_jsonl(file_path)
            
            # Convert None values to appropriate defaults and remove fields not in schema
            # Get the list of expected feature names
            expected_fields = set(features.keys())
            
            filtered_data = []
            for entry in data:
                # Only keep fields that are in the features schema
                filtered_entry = {k: v for k, v in entry.items() if k in expected_fields}
                
                # Convert None values to appropriate defaults
                for key in filtered_entry:
                    if filtered_entry[key] is None:
                        filtered_entry[key] = ""  # Default to empty string for string fields
                
                filtered_data.append(filtered_entry)
            
            splits[split_name] = Dataset.from_list(filtered_data, features=features)
            print(f"Loaded {len(filtered_data)} entries for {split_name}")
        else:
            print(f"Warning: {file_path} not found, skipping {split_name} split")
    
    if not splits:
        print("Error: No splits found. At least one of train.jsonl, validation.jsonl, or test.jsonl must exist.")
        return None
    
    return DatasetDict(splits)

def main():
    parser = argparse.ArgumentParser(description="Upload OpenForesight dataset to Hugging Face")
    parser.add_argument("--data_dir", default="/fast/nchandak/forecasting/openforesight", 
                       help="Directory containing standardized JSONL files")
    parser.add_argument("--repo_id", default="nikhilchandak/OpenForesight", 
                       help="Hugging Face repository ID")
    parser.add_argument("--token", help="Hugging Face token (optional, will use cached token if available)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("Creating dataset from files...")
    dataset_dict = create_dataset_dict(data_dir)
    
    if dataset_dict is None:
        print("Failed to create dataset. Exiting.")
        return
    
    print(f"Dataset created with splits: {list(dataset_dict.keys())}")
    print(f"Total entries: {sum(len(split) for split in dataset_dict.values())}")
    
    print("Uploading to Hugging Face...")
    try:
        # Push the dataset
        dataset_dict.push_to_hub(
            repo_id=args.repo_id,
            token=args.token,
            private=False
        )
        print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")
        
        # Push the README if it exists
        readme_path = data_dir / "README.md"
        if readme_path.exists():
            print("Uploading README.md...")
            api = HfApi(token=args.token)
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset"
            )
            print("Successfully uploaded README.md")
        else:
            print(f"Warning: README.md not found at {readme_path}")
            
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("Make sure you're logged in to Hugging Face with: huggingface-cli login")
        raise

if __name__ == "__main__":
    main()

