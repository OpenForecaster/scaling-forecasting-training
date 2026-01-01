#!/usr/bin/env python3
"""
FutureX Data Processing Script

Purpose:
    Downloads and processes FutureX-Online forecasting dataset from Hugging Face and Google Sheets.
    Converts data to JSONL format with prediction submissions.

Main Functions:
    - download_hf_dataset(): Downloads Futurex-Online dataset from HF hub
    - load_google_sheets(): Fetches data directly from Google Sheets
    - convert_csv_to_jsonl(): Converts CSV format to JSONL with id and prediction columns

Output:
    - futurex_predictions.jsonl: Prediction submissions for FutureX benchmark

Usage:
    python futureX.py
"""

from huggingface_hub import hf_hub_download
import pandas as pd
import os
import json
import requests

REPO_ID = "futurex-ai/Futurex-Online"
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1Qumh20Il_vzwQ1iBplMg8ZIMJK9EnPS5t-lDiGhl5Js/edit?gid=0#gid=0"

def download_hf_dataset():
    """Download the Futurex-Online dataset from Hugging Face"""
    print("Downloading Futurex-Online dataset from Hugging Face...")
    
    try:
        # Download the main data file (it's a parquet file, not JSON)
        data_file = hf_hub_download(
            repo_id=REPO_ID,
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset"
        )
        print(f"Downloaded: {data_file}")
        
        # Load the Parquet data
        df = pd.read_parquet(data_file)
        
        # Save as CSV
        output_file = "futurex_online.csv"
        df.to_csv(output_file, index=False)
        print(f"Dataset saved as CSV: {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Try alternative approach - list available files first
        try:
            from huggingface_hub import list_repo_files
            files = list_repo_files(REPO_ID, repo_type="dataset")
            print(f"Available files in repository: {files}")
        except Exception as e2:
            print(f"Error listing files: {e2}")

def load_google_sheets():
    """Load Google Sheets directly using the published CSV export URL"""
    print("Loading Google Sheets data...")
    
    try:
        # Convert the edit URL to a CSV export URL
        sheet_id = "1Qumh20Il_vzwQ1iBplMg8ZIMJK9EnPS5t-lDiGhl5Js"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
        
        print(f"Fetching data from: {csv_url}")
        
        # Download the CSV data
        response = requests.get(csv_url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the raw CSV data
        with open("futurex_sheets.csv", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        print("Successfully downloaded Google Sheets data as CSV")
        
        # Load and display the data
        df = pd.read_csv("futurex_sheets.csv")
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error loading Google Sheets: {e}")
        print("\nAlternative approach:")
        print("1. Open the Google Sheets in your browser")
        print("2. Go to File → Download → CSV")
        print("3. Save the file as 'futurex_sheets.csv' in this directory")
        print("4. Run the script again")
        return None

def convert_csv_to_jsonl(csv_file_path, output_jsonl_path):
    """Convert CSV file to JSONL format with only id and prediction columns"""
    print(f"Converting {csv_file_path} to JSONL format...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if required columns exist
        if 'id' not in df.columns:
            print("Error: 'id' column not found in CSV")
            print("Available columns:", list(df.columns))
            return
        
        if 'prediction' not in df.columns:
            print("Error: 'prediction' column not found in CSV")
            print("Available columns:", list(df.columns))
            return
        
        # Filter to only id and prediction columns
        df_filtered = df[['id', 'prediction']].copy()
        
        # Remove rows with missing values
        df_filtered = df_filtered.dropna()
        
        print(f"Filtered data shape: {df_filtered.shape}")
        print(f"Sample data:")
        print(df_filtered.head())
        
        # Convert to JSONL format
        with open(output_jsonl_path, 'w') as f:
            for _, row in df_filtered.iterrows():
                json_line = {
                    'id': row['id'],
                    'prediction': row['prediction']
                }
                f.write(json.dumps(json_line) + '\n')
        
        print(f"Successfully saved JSONL file: {output_jsonl_path}")
        print(f"Total lines written: {len(df_filtered)}")
        
    except Exception as e:
        print(f"Error converting CSV to JSONL: {e}")

def main():
    print("FutureX Data Processing Script")
    print("=" * 40)
    
    # # Option 1: Download from Hugging Face
    # print("\n1. Downloading from Hugging Face...")
    # download_hf_dataset()
    
    # Option 2: Load Google Sheets directly
    print("\n1. Loading Google Sheets data...")
    df = load_google_sheets()
    
    if df is not None:
        # Convert to JSONL format
        print("\n2. Converting to JSONL format...")
        convert_csv_to_jsonl("futurex_sheets.csv", "futurex_predictions.jsonl")
    else:
        # Option 3: Convert local CSV to JSONL
        print("\n2. Converting local CSV to JSONL...")
        csv_file = "futurex_sheets.csv"  # Change this to your CSV file path
        
        if os.path.exists(csv_file):
            convert_csv_to_jsonl(csv_file, "futurex_predictions.jsonl")
        else:
            print(f"CSV file {csv_file} not found. Please:")
            print("1. Export your Google Sheets as CSV")
            print("2. Save it in the data/ directory")
            print("3. Update the csv_file variable in the script")
            print("4. Run the script again")

if __name__ == "__main__":
    main()
