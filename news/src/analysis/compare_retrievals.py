#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import re
import random
from typing import Dict, List, Tuple, Set, Optional
from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm
import dateutil.parser
import datetime

def extract_dataset_params(dataset_name: str) -> Tuple[str, str, str, str]:
    """Extract parameters from dataset name."""
    parts = dataset_name.split('_')
    base_name = parts[0]
    source = parts[1]
    cutoff = parts[2]
    lookback = parts[3]
    
    return base_name, source, cutoff, lookback

def load_all_datasets(directory: str) -> Dict[str, any]:
    """Load all Hugging Face datasets from the given directory."""
    datasets = {}
    
    for path in Path(directory).glob("*"):
        if path.is_dir():
            try:
                dataset_name = path.name
                print(f"Loading dataset: {dataset_name}")
                dataset = load_from_disk(str(path))
                datasets[dataset_name] = dataset
                print(f"  - Loaded {len(dataset)} entries")
            except Exception as e:
                print(f"  - Error loading dataset {dataset_name}: {str(e)}")
    
    return datasets

def get_article_identifier(article: Dict) -> str:
    """Create a unique identifier for an article based on its URL and title."""
    url = article.get('url', '')
    title = article.get('title', '')
    return f"{url}_{title}"

def get_article_date(article: Dict) -> Optional[datetime.datetime]:
    """Get the most recent date from an article's date fields."""
    dates = []
    for date_field in ['date_publish', 'date_modify', 'date_download']:
        if date_field in article and article[date_field]:
            try:
                date = dateutil.parser.parse(article[date_field])
                # Convert to naive datetime by removing timezone info
                if date.tzinfo is not None:
                    date = date.replace(tzinfo=None)
                if date:
                    dates.append(date)
            except (ValueError, TypeError):
                continue
    
    return max(dates) if dates else None

def compare_article_lists(list1: List[Dict], list2: List[Dict]) -> Tuple[int, Set[str], Set[str], float]:
    """
    Compare two lists of articles and return information about differences.
    
    Returns:
        - Number of common articles
        - Set of article IDs unique to list1
        - Set of article IDs unique to list2
        - Average position change for common articles
    """
    # Create sets of article identifiers and position mappings
    ids1 = {get_article_identifier(article) for article in list1}
    ids2 = {get_article_identifier(article) for article in list2}
    
    # Create position mappings (article_id -> position)
    pos_map1 = {get_article_identifier(article): i for i, article in enumerate(list1)}
    pos_map2 = {get_article_identifier(article): i for i, article in enumerate(list2)}
    
    # Find unique articles
    unique_to_list1 = ids1 - ids2
    unique_to_list2 = ids2 - ids1
    
    # Calculate position changes for common articles
    common_articles = ids1.intersection(ids2)
    position_changes = []
    
    for article_id in common_articles:
        pos1 = pos_map1[article_id]
        pos2 = pos_map2[article_id]
        position_changes.append(abs(pos1 - pos2))
    
    # Calculate average position change
    avg_position_change = sum(position_changes) / len(position_changes) if position_changes else 0
    
    return len(common_articles), unique_to_list1, unique_to_list2, avg_position_change

def run_comparison(datasets: Dict[str, any], date_filter: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """Run comparison between all dataset pairs and return a DataFrame with results."""
    results = []
    
    # Get all dataset names
    dataset_names = list(datasets.keys())
    
    # Create all possible pairs
    for i in range(len(dataset_names)):
        for j in range(i+1, len(dataset_names)):
            name1 = dataset_names[i]
            name2 = dataset_names[j]
            
            ds1 = datasets[name1]
            ds2 = datasets[name2]
            
            # Extract parameters
            base1, source1, cutoff1, lookback1 = extract_dataset_params(name1)
            base2, source2, cutoff2, lookback2 = extract_dataset_params(name2)
            
            # Only compare datasets with same base name and source
            if base1 != base2 or source1 != source2:
                continue
                
            # If datasets have different lengths, use the smaller one
            common_length = min(len(ds1), len(ds2))
            
            print(f"Comparing {name1} vs {name2}")
            
            # Track statistics
            total_questions = 0
            questions_with_different_articles = 0
            questions_with_reranking = 0
            total_reranked_articles = 0
            sum_position_changes = 0
            
            # Track date differences
            date_diffs_ds1 = []
            date_diffs_ds2 = []
            
            # Store sample questions with differences
            different_article_samples = []
            
            # Iterate through questions
            for q_idx in tqdm(range(common_length), desc="Comparing questions"):
                if "retrieved_articles" not in ds1[q_idx] or "retrieved_articles" not in ds2[q_idx]:
                    continue
                
                # Apply date filter if specified
                if date_filter:
                    start_year, end_year = date_filter
                    if "date_resolve_at" not in ds1[q_idx] or not ds1[q_idx]["date_resolve_at"]:
                        continue
                    
                    try:
                        resolve_date = dateutil.parser.parse(ds1[q_idx]["date_resolve_at"])
                        # Make resolution date timezone-naive if it has timezone info
                        if resolve_date.tzinfo is not None:
                            resolve_date = resolve_date.replace(tzinfo=None)
                        if resolve_date.year < start_year or resolve_date.year > end_year:
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                articles1 = ds1[q_idx]["retrieved_articles"]
                articles2 = ds2[q_idx]["retrieved_articles"]
                
                # Skip if either has no articles
                if not articles1 or not articles2:
                    continue
                
                total_questions += 1
                
                # Compare article lists
                common_count, unique_to_1, unique_to_2, avg_position_change = compare_article_lists(articles1, articles2)
                
                # Calculate date differences if resolution date is available
                if "date_resolve_at" in ds1[q_idx] and ds1[q_idx]["date_resolve_at"]:
                    try:
                        resolve_date = dateutil.parser.parse(ds1[q_idx]["date_resolve_at"])
                        # Make resolution date timezone-naive if it has timezone info
                        if resolve_date.tzinfo is not None:
                            resolve_date = resolve_date.replace(tzinfo=None)
                        
                        # Calculate date differences for dataset 1
                        for article in articles1:
                            article_date = get_article_date(article)
                            if article_date and resolve_date:
                                date_diff = (resolve_date - article_date).days
                                date_diffs_ds1.append(date_diff)
                        
                        # Calculate date differences for dataset 2
                        for article in articles2:
                            article_date = get_article_date(article)
                            if article_date and resolve_date:
                                date_diff = (resolve_date - article_date).days
                                date_diffs_ds2.append(date_diff)
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing date for question {q_idx}: {e}")
                        pass
                
                # Update statistics
                if unique_to_1 or unique_to_2:
                    questions_with_different_articles += 1
                    
                    # Collect sample if interesting (has differences)
                    different_article_samples.append({
                        "q_idx": q_idx,
                        "question": ds1[q_idx].get("question", ""),
                        "unique_to_1": list(unique_to_1),
                        "unique_to_2": list(unique_to_2),
                        "articles1": articles1,
                        "articles2": articles2
                    })
                
                if common_count > 0 and avg_position_change > 0:
                    questions_with_reranking += 1
                    total_reranked_articles += common_count
                    sum_position_changes += avg_position_change
            
            # Calculate average position change across all questions
            avg_position_change_overall = sum_position_changes / questions_with_reranking if questions_with_reranking > 0 else 0
            
            # Calculate mean date differences
            mean_date_diff_ds1 = sum(date_diffs_ds1) / len(date_diffs_ds1) if date_diffs_ds1 else 0
            mean_date_diff_ds2 = sum(date_diffs_ds2) / len(date_diffs_ds2) if date_diffs_ds2 else 0
            
            # Add comparison results to results list
            results.append({
                "cutoff1": cutoff1,
                "cutoff2": cutoff2,
                "lookback1": lookback1,
                "lookback2": lookback2,
                "total_questions": total_questions,
                "questions_with_different_articles_pct": questions_with_different_articles / total_questions * 100 if total_questions > 0 else 0,
                "questions_with_reranking_pct": questions_with_reranking / total_questions * 100 if total_questions > 0 else 0,
                "total_reranked_articles": total_reranked_articles,
                "avg_position_change": avg_position_change_overall,
                "mean_date_diff_ds1": mean_date_diff_ds1,
                "mean_date_diff_ds2": mean_date_diff_ds2,
                "samples": different_article_samples
            })
    
    return pd.DataFrame(results)

def print_samples(results_df: pd.DataFrame, output_dir: str):
    """Print all questions with different articles to separate files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each comparison
    for idx, row in results_df.iterrows():
        samples = row["samples"]
        if not samples:
            continue
        
        # Create filename based on cutoffs and lookbacks
        filename = f"{output_dir}/comparison_cutoff{row['cutoff1']}_cutoff{row['cutoff2']}_lookback{row['lookback1']}_lookback{row['lookback2']}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"\n--- Comparison: cutoff1={row['cutoff1']}, cutoff2={row['cutoff2']}, lookback1={row['lookback1']}, lookback2={row['lookback2']} ---\n")
            
            # Print all samples
            for i, sample in enumerate(samples):
                f.write(f"\nSample {i+1}: Question {sample['q_idx']}\n")
                f.write(f"Question: {sample['question']}\n")
                
                f.write("\nArticles unique to first dataset:\n")
                for article_id in sample["unique_to_1"]:
                    # Find the article with this ID
                    for article in sample["articles1"]:
                        if get_article_identifier(article) == article_id:
                            f.write(f"  - Title: {article.get('title', 'N/A')}\n")
                            f.write(f"    URL: {article.get('url', 'N/A')}\n")
                            f.write(f"    Date: {article.get('date_publish', 'N/A')}\n")
                            f.write(f"    Score: {article.get('score', 'N/A')}\n")
                            break
                
                f.write("\nArticles unique to second dataset:\n")
                for article_id in sample["unique_to_2"]:
                    # Find the article with this ID
                    for article in sample["articles2"]:
                        if get_article_identifier(article) == article_id:
                            f.write(f"  - Title: {article.get('title', 'N/A')}\n")
                            f.write(f"    URL: {article.get('url', 'N/A')}\n")
                            f.write(f"    Date: {article.get('date_publish', 'N/A')}\n")
                            f.write(f"    Score: {article.get('score', 'N/A')}\n")
                            break
                
                f.write("\n" + "-"*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Compare retrieved articles across datasets")
    parser.add_argument("--directory", type=str, 
                        default="/is/cluster/fast/sgoel/forecasting/news/retrieval",
                        help="Directory containing Hugging Face datasets")
    parser.add_argument("--output", type=str, default="analysis_results/retrieval_comparison/",
                        help="Output directory for comparison results")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of sample questions to show for each comparison")
    parser.add_argument("--date-filter", type=str, 
                        help="Filter questions by resolution date range (format: YYYY-YYYY)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Parse date filter if provided
    date_filter = None
    if args.date_filter:
        try:
            start_year, end_year = map(int, args.date_filter.split('-'))
            date_filter = (start_year, end_year)
            print(f"Filtering questions with resolution dates between {start_year} and {end_year}")
        except ValueError:
            print(f"Invalid date filter format: {args.date_filter}. Expected format: YYYY-YYYY")
            return
    
    # Load all datasets
    print(f"Loading datasets from {args.directory}")
    datasets = load_all_datasets(args.directory)
    
    if not datasets:
        print("No datasets found!")
        return
    
    # Run comparison
    print("Running comparisons between dataset pairs...")
    results_df = run_comparison(datasets, date_filter)
    
    # Sort results by lookback1, lookback2, and abs(cutoff1 - cutoff2)
    results_df['cutoff_diff'] = results_df.apply(lambda row: abs(int(row['cutoff1']) - int(row['cutoff2'])), axis=1)
    # Convert lookback values to integers for proper numerical sorting
    results_df['lookback1_int'] = results_df['lookback1'].astype(int)
    results_df['lookback2_int'] = results_df['lookback2'].astype(int)
    results_df = results_df.sort_values(by=['lookback1_int', 'lookback2_int', 'cutoff_diff'])
    # Drop temporary columns used for sorting
    results_df = results_df.drop(columns=['cutoff_diff', 'lookback1_int', 'lookback2_int'])
    
    # Save results to CSV
    results_df_for_csv = results_df.copy()
    # Remove samples column which has complex objects
    results_df_for_csv = results_df_for_csv.drop(columns=["samples"])
    
    # Add date filter info to filename if applicable
    date_filter_str = f"_filtered_{date_filter[0]}-{date_filter[1]}" if date_filter else ""
    csv_path = os.path.join(args.output, f"retrieval_comparison_results{date_filter_str}.csv")
    results_df_for_csv.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Print summary
    print("\n===== SUMMARY OF COMPARISONS =====")
    for idx, row in results_df.iterrows():
        print(f"\nComparing cutoff1={row['cutoff1']}, cutoff2={row['cutoff2']}, lookback1={row['lookback1']}, lookback2={row['lookback2']}:")
        print(f"  - Total questions analyzed: {row['total_questions']}")
        print(f"  - Questions with different articles: {row['questions_with_different_articles_pct']:.2f}%")
        print(f"  - Questions with reranked articles: {row['questions_with_reranking_pct']:.2f}%")
        print(f"  - Average position change for reranked articles: {row['avg_position_change']:.2f}")
        print(f"  - Mean days between question resolution and articles (dataset 1): {row['mean_date_diff_ds1']:.2f}")
        print(f"  - Mean days between question resolution and articles (dataset 2): {row['mean_date_diff_ds2']:.2f}")
    
    # Print samples to separate files
    samples_dir = f"{args.output}/samples{date_filter_str}"
    print_samples(results_df, samples_dir)

if __name__ == "__main__":
    main()
