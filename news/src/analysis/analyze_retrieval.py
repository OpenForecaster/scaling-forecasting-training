import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
import random
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description="Plot retrieval trends from BM25 retrieval")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save plots")
    parser.add_argument("--dataset_dirs", type=str, nargs='+', required=True,
                        help="List of directories containing datasets and trends.pkl files")
    parser.add_argument("--sample_percent", type=float, default=1.0,
                        help="Percentage of dataset rows to sample (default: 1%)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_datasets_df = []
    
    # Process each dataset directory
    for dataset_dir in args.dataset_dirs:
        # Extract dataset label from the last subfolder
        dataset_label = os.path.basename(os.path.normpath(dataset_dir))
        
        # Load trend data from the dataset directory
        trend_data_path = os.path.join(dataset_dir, "trends.pkl")
        with open(trend_data_path, 'rb') as f:
            trend_data = pickle.load(f)
        
        # Load the dataset using Hugging Face's load_from_disk
        dataset = load_from_disk(dataset_dir)
        
        # Print random samples from the dataset if sample_percent > 0
        if args.sample_percent > 0:
            sample_size = int(len(dataset) * args.sample_percent / 100)
            sample_size = max(1, sample_size)  # Ensure at least one sample
            
            print(f"\n===== Sampling {args.sample_percent}% ({sample_size} examples) from dataset {dataset_label} =====\n")
            
            # Get random indices
            random_indices = random.sample(range(len(dataset)), sample_size)
            
            for idx in random_indices:
                example = dataset[idx]
                print(f"Example {idx}:")
                print(f"Question: {example['question']}")
                print(f"Date Resolved: {example['date_resolve_at']}")
                print(f"Retrieved Articles:")
                for i, article in enumerate(example['retrieved_articles'][:2]):
                    print(f"  {i+1}. Title: {article.get('title', 'No title')}")
                    print(f"     Date Published: {article.get('date_publish', 'Unknown')}")
                print("-" * 80)
        
        # Convert trend data to DataFrame
        data_list = []
        for question_id, values in trend_data.items():
            entry = {
                'question_id': question_id,
                'date': values['date'],
                'year': values['date'].year,
                'month': values['date'].month,
                'index_size': values['index_size'],
                'highest_score': values['highest_score'],
                'mean_score': values['mean_score'],
                'num_retrieved': values['num_retrieved'],
                'dataset': dataset_label  # Add dataset label
            }
            data_list.append(entry)
        
        df = pd.DataFrame(data_list)
        all_datasets_df.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_datasets_df, ignore_index=True)
    
    # Create year-month column for proper time-series plotting
    combined_df['year_month'] = combined_df['date'].dt.to_period('M')
    
    # Define metrics to plot
    metrics = {
        'index_size': {
            'title': 'Average Index Size by Month',
            'ylabel': 'Average Number of Articles in Index',
            'color': 'blue',
            'filename': 'index_size_by_month.png'
        },
        'highest_score': {
            'title': 'Average Highest Score by Month',
            'ylabel': 'Average Score of Top Article',
            'color': 'green',
            'filename': 'highest_score_by_month.png'
        },
        'mean_score': {
            'title': 'Average Mean Score by Month',
            'ylabel': 'Average of Mean Scores of Top 5 Articles',
            'color': 'purple',
            'filename': 'mean_score_by_month.png'
        },
        'num_retrieved': {
            'title': 'Average Number of Retrieved Articles by Month',
            'ylabel': 'Average Number of Retrieved Articles',
            'color': 'red',
            'filename': 'num_retrieved_by_month.png'
        }
    }
    
    # Plot each metric with multiple datasets
    for metric, config in metrics.items():
        plt.figure(figsize=(14, 7))
        
        # Get unique datasets
        datasets = combined_df['dataset'].unique()
        
        # Use a colormap to assign different colors to each dataset
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        for i, dataset_name in enumerate(datasets):
            df_subset = combined_df[combined_df['dataset'] == dataset_name]
            df_grouped = df_subset.groupby('year_month')[metric].mean().reset_index()
            
            # Sort by year_month for proper time series
            df_grouped = df_grouped.sort_values('year_month')
            
            plt.plot(df_grouped['year_month'].astype(str), df_grouped[metric], 
                     marker='o', linewidth=2, color=colors[i], label=dataset_name)
        
        plt.title(config['title'])
        plt.xlabel('Year-Month')
        plt.ylabel(config['ylabel'])
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, config['filename']))
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()