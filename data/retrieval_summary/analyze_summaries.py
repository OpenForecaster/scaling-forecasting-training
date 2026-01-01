"""
Script for analyzing and comparing generated article summaries.
"""

import os
import pandas as pd
import datasets
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
import argparse
import logging
from typing import List, Dict, Any, Optional
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(re.findall(r'\b\w+\b', text))

class SummaryAnalyzer:
    """
    Class for analyzing article summaries generated using different prompts.
    """
    
    def __init__(self, summaries_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            summaries_dir (str): Directory containing the summary datasets
        """
        self.summaries_dir = summaries_dir
        self.datasets = {}
        self.prompt_stats = defaultdict(dict)
        
    def load_datasets(self):
        """
        Load all summary datasets in the directory.
        """
        # Find all dataset files
        pattern = os.path.join(self.summaries_dir, "*.dataset")
        dataset_paths = glob.glob(pattern)
        
        if not dataset_paths:
            logger.error(f"No dataset files found in {self.summaries_dir}")
            return
        
        logger.info(f"Found {len(dataset_paths)} dataset files")
        
        # Load each dataset
        for path in dataset_paths:
            name = os.path.basename(path)
            logger.info(f"Loading dataset: {name}")
            
            try:
                dataset = datasets.load_from_disk(path)
                self.datasets[name] = dataset
                logger.info(f"Loaded dataset with {len(dataset)} rows")
            except Exception as e:
                logger.error(f"Error loading dataset {path}: {e}")
    
    def analyze_summary_lengths(self):
        """
        Analyze the lengths of the summaries.
        """
        results = []
        
        for name, dataset in self.datasets.items():
            # Extract prompt name and target length from the filename
            match = re.match(r'(\w+)_length(\d+)\.dataset', name)
            if not match:
                logger.warning(f"Could not parse filename: {name}")
                continue
                
            prompt_name, target_length = match.groups()
            target_length = int(target_length)
            
            # Collect word counts for all summaries
            word_counts = []
            
            for row in dataset:
                for summary_item in row["articles_summary"]:
                    if summary_item and "summary" in summary_item:
                        word_count = count_words(summary_item["summary"])
                        word_counts.append(word_count)
            
            if not word_counts:
                logger.warning(f"No summaries found in dataset: {name}")
                continue
                
            # Calculate statistics
            count = len(word_counts)
            mean = np.mean(word_counts)
            std = np.std(word_counts)
            median = np.median(word_counts)
            min_count = np.min(word_counts)
            max_count = np.max(word_counts)
            
            # Store statistics
            self.prompt_stats[(prompt_name, target_length)] = {
                "count": count,
                "mean": mean,
                "std": std,
                "median": median,
                "min": min_count,
                "max": max_count,
                "target": target_length
            }
            
            # Add to results
            results.append({
                "prompt_name": prompt_name,
                "target_length": target_length,
                "count": count,
                "mean": mean,
                "std": std,
                "median": median,
                "min": min_count,
                "max": max_count
            })
        
        # Create a DataFrame
        return pd.DataFrame(results)
    
    def plot_summary_lengths(self, results_df):
        """
        Plot the summary lengths.
        
        Args:
            results_df (pd.DataFrame): DataFrame with summary length statistics
        """
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Group by prompt name
        for prompt_name, group in results_df.groupby("prompt_name"):
            # Sort by target length
            group = group.sort_values("target_length")
            
            # Plot
            plt.plot(
                group["target_length"], 
                group["mean"], 
                marker='o', 
                label=prompt_name
            )
            
            # Add error bars
            plt.errorbar(
                group["target_length"], 
                group["mean"], 
                yerr=group["std"], 
                alpha=0.3, 
                fmt='none'
            )
        
        # Add target line
        target_lengths = sorted(results_df["target_length"].unique())
        plt.plot(target_lengths, target_lengths, 'k--', label="Target")
        
        plt.xlabel("Target Length (words)")
        plt.ylabel("Actual Length (words)")
        plt.title("Summary Lengths by Prompt Type and Target Length")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        output_path = os.path.join(self.summaries_dir, "summary_lengths.png")
        plt.savefig(output_path)
        logger.info(f"Saved plot to {output_path}")
        
        # Show the figure
        plt.show()
    
    def sample_summaries(self, n_samples=3):
        """
        Sample summaries from each prompt type and target length.
        
        Args:
            n_samples (int): Number of samples to show for each prompt type and target length
            
        Returns:
            pd.DataFrame: DataFrame with sample summaries
        """
        samples = []
        
        for name, dataset in self.datasets.items():
            # Extract prompt name and target length from the filename
            match = re.match(r'(\w+)_length(\d+)\.dataset', name)
            if not match:
                continue
                
            prompt_name, target_length = match.groups()
            target_length = int(target_length)
            
            # Get a few random rows
            if len(dataset) > 0:
                indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
                
                for idx in indices:
                    # Convert numpy.int64 to Python int to avoid indexing error
                    row = dataset[int(idx)]
                    
                    # Get a random article summary
                    article_summaries = [s for s in row["articles_summary"] if s and "summary" in s]
                    if article_summaries:
                        summary_item = np.random.choice(article_summaries)
                        
                        samples.append({
                            "prompt_name": prompt_name,
                            "target_length": target_length,
                            "question": row["question"],
                            "summary": summary_item["summary"],
                            "word_count": count_words(summary_item["summary"])
                        })
        
        # Create a DataFrame
        return pd.DataFrame(samples)

def main():
    """
    Main function to run the analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze article summaries")
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default="/fast/nchandak/forecasting/retrieval_summary/metaculus-binary_reuters_7_365",
        help="Directory containing the summary datasets"
    )
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = SummaryAnalyzer(args.summaries_dir)
    
    # Load the datasets
    analyzer.load_datasets()
    
    # Analyze summary lengths
    results_df = analyzer.analyze_summary_lengths()
    if results_df is not None and not results_df.empty:
        # Print the results
        print("\nSummary Length Statistics:")
        print(results_df.sort_values(["prompt_name", "target_length"]).to_string(index=False))
        
        # Plot the results
        analyzer.plot_summary_lengths(results_df)
    
    # Sample summaries
    samples_df = analyzer.sample_summaries(n_samples=1)
    if samples_df is not None and not samples_df.empty:
        # Save the samples to a CSV file
        output_path = os.path.join(args.summaries_dir, "sample_summaries.csv")
        samples_df.to_csv(output_path, index=False)
        logger.info(f"Saved sample summaries to {output_path}")
        
        # Print a few samples
        print("\nSample Summaries:")
        for _, sample in samples_df.iterrows():
            print(f"\nPrompt: {sample['prompt_name']}, Target Length: {sample['target_length']}")
            print(f"Question: {sample['question']}")
            print(f"Summary ({sample['word_count']} words): {sample['summary']}")
            print("-" * 80)

if __name__ == "__main__":
    main() 