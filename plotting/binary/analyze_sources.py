#!/usr/bin/env python3
"""
Analyze news sources from Metaculus JSONL data and create a bar plot.

This script reads a JSONL file containing Metaculus questions with retrieved articles,
extracts the news sources (domains), counts their occurrences across all questions,
and creates a horizontal bar plot showing the total number of articles retrieved
per news source.
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import sys

def load_data(file_path):
    """Load data from JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} questions from {file_path}")
    return data

def extract_news_sources(data):
    """Extract news sources from relevant_articles_sorted_by_docs field."""
    source_counter = Counter()
    
    for question in data:
        relevant_articles = question.get('relevant_articles_sorted_by_docs', [])
        
        if relevant_articles:
            for article in relevant_articles:
                # Each article is a list: [score, hash, article_data]
                if len(article) >= 3 and isinstance(article[2], dict):
                    article_data = article[2]
                    source_domain = article_data.get('source_domain', 'Unknown')
                    source_counter[source_domain] += 1
    
    print(f"Found {len(source_counter)} unique news sources")
    print(f"Total articles across all questions: {sum(source_counter.values())}")
    
    return source_counter

def create_bar_plot(source_counter, output_path):
    """Create and save horizontal bar plot of news sources."""
    if not source_counter:
        print("No news sources found in the data.")
        return
    
    # Sort sources by count (descending)
    sorted_sources = source_counter.most_common()
    
    # Keep only the top 10 sources
    sorted_sources = sorted_sources[:10]
    
    # Prepare data for plotting
    sources = [item[0] for item in sorted_sources]
    counts = [item[1] for item in sorted_sources]
    
    # Create figure and axis
    plt.figure(figsize=(12, max(8, len(sources) * 0.3)))
    
    # Set style
    sns.set_style("whitegrid")
    colors = plt.cm.viridis(range(len(sources)))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(sources)), counts, color=colors)
    
    # Customize the plot
    plt.xlabel('Total Number of Articles Retrieved', fontsize=12, fontweight='bold')
    plt.ylabel('News Source', fontsize=12, fontweight='bold')
    plt.title('News Sources Distribution in Retrieved Articles\n(Sorted by Article Count)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis labels (news sources)
    plt.yticks(range(len(sources)), sources)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                str(count), ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Invert y-axis so highest counts are at the top
    plt.gca().invert_yaxis()
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    
    # Print top 10 sources for verification
    print("\nTop 10 news sources by article count:")
    for i, (source, count) in enumerate(sorted_sources[:10], 1):
        print(f"{i:2d}. {source:<30} {count:>4d} articles")

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze news sources from Metaculus JSONL data and create a bar plot."
    )
    parser.add_argument(
        'input_file', 
        nargs='?',
        default='/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/with_retreival/metaculus-05-2025_30.jsonl',
        help='Path to input JSONL file (default: %(default)s)'
    )
    parser.add_argument(
        '--output-dir',
        default='plots/news_sources/',
        help='Output directory for the plot (default: %(default)s)'
    )
    parser.add_argument(
        '--filename',
        default='news_sources_distribution.png',
        help='Output filename (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
    
    # Set output path
    output_path = Path(args.output_dir) / args.filename
    
    print(f"Input file: {input_path}")
    print(f"Output path: {output_path}")
    print("-" * 50)
    
    # Load and analyze data
    data = load_data(input_path)
    source_counter = extract_news_sources(data)
    
    # Create and save plot
    create_bar_plot(source_counter, output_path)
    
    print("-" * 50)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
