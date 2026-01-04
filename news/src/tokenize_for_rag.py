import json
import os
import re
import argparse
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from datasets import load_dataset, Dataset

# Uncomment these lines for first run
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize and preprocess text using NLTK.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of preprocessed tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove non-alphabetic characters and filter short words
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 2]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def tokenize_news_articles(jsonl_path: str, output_path: str, delete_original: bool = False) -> None:
    """
    Tokenize news articles from JSONL file and save with tokenized content.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to save tokenized JSONL file
        delete_original: Whether to delete the original file after tokenizing
    """
    # Load articles from input file
    articles = []
    print(f"Loading articles from {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line)
                articles.append(article)
            except json.JSONDecodeError:
                continue
    
    print(f"Tokenizing {len(articles)} articles")
    
    for article in tqdm(articles, desc="Tokenizing news articles"):
        # Combine title and maintext for tokenization
        title = article.get('title', '')
        maintext = article.get('maintext', '')
        combined_text = f"{title} {maintext}"
        
        # Tokenize and add to article
        article['tokenized_news'] = tokenize_text(combined_text)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Append or write tokenized articles
    mode = 'a' if os.path.exists(output_path) else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article) + '\n')
    
    print(f"{'Appended' if mode == 'a' else 'Saved'} tokenized news to {output_path}")
    
    # Delete original file if requested
    if delete_original and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print(f"Deleted original file: {jsonl_path}")

def tokenize_metaculus_questions(dataset_path: str, output_path: str) -> None:
    """
    Tokenize Metaculus questions and save with tokenized content.
    
    Args:
        dataset_path: Path or name of Metaculus dataset
        output_path: Path to save tokenized dataset
    """
    # Check if output directory already exists
    if os.path.exists(output_path):
        print(f"Output directory {output_path} already exists. Skipping processing.")
        return
    
    print(f"Loading Metaculus dataset from {dataset_path}")
    ds = load_dataset(dataset_path)["train"]
    
    # Tokenize questions
    tokenized_queries = []
    for i in tqdm(range(len(ds)), desc="Tokenizing Metaculus questions"):
        question = ds['question'][i]
        tokenized_query = tokenize_text(question)
        tokenized_queries.append(tokenized_query)
    
    # Add tokenized column to dataset
    new_ds = ds.add_column("tokenized_query", tokenized_queries)
    
    # Save tokenized dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_ds.save_to_disk(output_path)
    print(f"Saved tokenized Metaculus questions to {output_path}")

def process_jsonl_file(args_tuple: Tuple[str, str, bool]) -> None:
    """Process a single JSONL file (for parallel processing)"""
    jsonl_path, output_path, delete_original = args_tuple
    tokenize_news_articles(jsonl_path, output_path, delete_original)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tokenize news articles and Metaculus questions")
    
    parser.add_argument("--jsonl_path", type=str, default="/fast/sgoel/forecasting/news/filtered_cc_articles/jsonl/apnews.com.jsonl", 
                        help="Path to a JSONL file or directory containing JSONL files")
    parser.add_argument("--metaculus_dataset", type=str, default="nikhilchandak/metaculus-binary",
                        help="Path or name of the Metaculus dataset")
    parser.add_argument("--output_dir", type=str, default="/fast/sgoel/forecasting/news/tokenized_data/",
                        help="Directory to save tokenized outputs")
    parser.add_argument("--num_workers", type=int, default=48,
                        help="Number of workers for parallel processing")
    parser.add_argument("--delete_original", action="store_true",
                        help="Delete original files after tokenizing")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Process news articles
    jsonl_paths = []
    if args.jsonl_path:
        if os.path.isdir(args.jsonl_path):
            # If it's a directory, get all JSONL files
            jsonl_paths = [os.path.join(args.jsonl_path, f) for f in os.listdir(args.jsonl_path) 
                          if f.endswith('.jsonl')]
        elif os.path.isfile(args.jsonl_path):
            # If it's a single file
            jsonl_paths = [args.jsonl_path]
    
    # Prepare arguments for parallel processing
    parallel_args = []
    for jsonl_path in jsonl_paths:
        file_name = os.path.basename(jsonl_path)
        base_name = file_name
        if file_name.endswith('.jsonl'):
            base_name = file_name[:-6]  # Remove .jsonl extension
        output_path = os.path.join(args.output_dir, "news", f"{base_name}_tokenized.jsonl")
        parallel_args.append((jsonl_path, output_path, args.delete_original))
    
    # Process files in parallel
    if parallel_args:
        os.makedirs(args.output_dir, exist_ok=True)
        if len(parallel_args) == 1 or args.num_workers <= 1:
            # Single file or single worker mode
            for arg_tuple in parallel_args:
                process_jsonl_file(arg_tuple)
        else:
            # Parallel processing
            print(f"Processing {len(parallel_args)} files with {args.num_workers} workers")
            with multiprocessing.Pool(processes=min(args.num_workers, len(parallel_args))) as pool:
                pool.map(process_jsonl_file, parallel_args)
    
    # Process Metaculus questions
    if args.metaculus_dataset:
        dataset_name = args.metaculus_dataset.split('/')[-1]
        output_path = os.path.join(args.output_dir, "questions", f"{dataset_name}_tokenized")
        tokenize_metaculus_questions(args.metaculus_dataset, output_path)

if __name__ == "__main__":
    main()