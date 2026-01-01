#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any
from urllib.parse import urlparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

mpl.style.use(['science'])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot percentage of questions passing human filter by news source"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_manualFilter.jsonl",
        help="Path to questions JSONL file",
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="plots",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum number of samples for a source to be included in the plot",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="human_filter",
        help="Field name to use for filtering (default: human_filter)",
    )
    return parser.parse_args()


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return "unknown"
    parsed = urlparse(
        url if (url.startswith('http://') or url.startswith('https://')) 
        else f"http://{url}"
    )
    host = parsed.netloc
    if not host:
        return "unknown"
    host = host.split(':')[0]
    if host.startswith('www.'):
        host = host[4:]
    return host.lower()


def calculate_source_statistics(data: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for each news source.
    
    Args:
        data: List of question dictionaries
        field: Field name to check for filtering (e.g., 'human_filter', 'model_filter')
    
    Returns:
        Dictionary mapping source domain to stats including:
        - total: total questions for this source
        - passed: number of questions with field = 1
        - percentage: percentage that passed filtering
    """
    source_stats = defaultdict(lambda: {'total': 0, 'passed': 0})
    
    for item in data:
        # Skip items without the specified field
        if field not in item:
            continue
        
        # Extract domain from article_url
        # domain = extract_domain(item.get('article_url', ''))
        domain = item.get('news_source', 'unknown')
        if "unknown" not in domain:
            domain = f"www.{domain}.com"
        
        # Update statistics
        source_stats[domain]['total'] += 1
        if item.get(field, 0) == 1:
            source_stats[domain]['passed'] += 1
    
    # Calculate percentages
    for domain in source_stats:
        total = source_stats[domain]['total']
        passed = source_stats[domain]['passed']
        source_stats[domain]['percentage'] = (passed / total * 100) if total > 0 else 0
    
    return dict(source_stats)


def plot_source_filtering_rates(
    source_stats: Dict[str, Dict[str, Any]],
    output_path: str,
    total_questions: int,
    min_samples: int = 5,
    field: str = "human_filter"
):
    """Plot the percentage of questions passing filter by news source."""
    
    # Filter sources by minimum sample size
    filtered_stats = {
        domain: stats for domain, stats in source_stats.items()
        if stats['total'] >= min_samples
    }
    
    if not filtered_stats:
        print(f"No sources with at least {min_samples} samples found")
        return
    
    # Sort by percentage (descending)
    sorted_sources = sorted(
        filtered_stats.items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    
    domains = [item[0] for item in sorted_sources]
    percentages = [item[1]['percentage'] for item in sorted_sources]
    totals = [item[1]['total'] for item in sorted_sources]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x_positions = np.arange(len(domains))
    bars = ax.bar(x_positions, percentages, alpha=0.85, color='#2ca02c')
    
    # Customize the plot
    ax.set_ylabel('Pass Rate (\%)', fontsize=26, fontweight='bold')
    # ax.set_xlabel('News Source', fontsize=26, fontweight='bold')
    
    # Set title with total question count
    field_display = field.replace('_', ' ').title()
    ax.set_title(
        f'{field_display} Pass Rate by News Source \ (Total Questions with Filter: {total_questions})',
        fontsize=24,
        fontweight='bold',
        pad=20
    )
    
    # Format domain names for display
    display_domains = []
    for domain in domains:
        display_name = domain
        display_domains.append(display_name)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_domains, rotation=45, ha='right', fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, percentage, total in zip(bars, percentages, totals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{percentage:.1f}\%', #\n(n={total})',
            ha='center',
            va='bottom',
            fontsize=24,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {pdf_path}")
    
    plt.close()


def main():
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    print(f"Processing file: {args.input_file}")
    
    # Load data
    data = load_jsonl_file(args.input_file)
    print(f"Loaded {len(data)} total rows")
    
    # Filter to only rows with the specified field
    filtered_data = [item for item in data if args.field in item]
    total_with_filter = len(filtered_data)
    print(f"Found {total_with_filter} rows with '{args.field}' field")
    
    if total_with_filter == 0:
        print(f"No rows with '{args.field}' field found")
        return
    
    # Calculate statistics by source
    source_stats = calculate_source_statistics(filtered_data, args.field)
    print(f"\nFound {len(source_stats)} unique news sources")
    
    # Print summary statistics
    print("\nSource Statistics:")
    for domain, stats in sorted(source_stats.items(), key=lambda x: x[1]['percentage'], reverse=True):
        print(f"  {domain}: {stats['passed']}/{stats['total']} passed ({stats['percentage']:.1f}%)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    input_filename = os.path.basename(args.input_file).replace('.jsonl', '')
    output_filename = f"source_filtering_{input_filename}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Create the plot
    plot_source_filtering_rates(
        source_stats,
        output_path,
        total_with_filter,
        args.min_samples,
        args.field
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

