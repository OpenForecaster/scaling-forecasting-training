import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def parse_date(date_str):
    """Parse date string to datetime object."""
    if not date_str:
        return None
    
    # Handle different date formats
    formats = [
        "%Y-%m-%d %H:%M:%S%z",  # 2022-10-14 18:25:58+00:00
        "%Y-%m-%d %H:%M:%S",     # 2022-10-14 00:00:00
        "%Y-%m-%d"               # 2022-10-14
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def process_jsonl_files(input_path):
    """Process JSONL file(s) from the given path."""
    # Statistics containers
    articles_by_month = defaultdict(int)
    articles_by_year = defaultdict(int)
    articles_by_language = defaultdict(int)
    articles_by_source = defaultdict(int)
    articles_by_source_month = defaultdict(lambda: defaultdict(int))
    total_articles = 0
    
    # Check if input is a directory or a single file
    if os.path.isdir(input_path):
        # Process each JSONL file in the directory
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(input_path, filename)
                total_articles = process_single_jsonl(
                    file_path, 
                    articles_by_month, 
                    articles_by_year,
                    articles_by_language, 
                    articles_by_source,
                    articles_by_source_month,
                    total_articles
                )
    else:
        # Process a single JSONL file
        total_articles = process_single_jsonl(
            input_path, 
            articles_by_month, 
            articles_by_year,
            articles_by_language, 
            articles_by_source,
            articles_by_source_month,
            total_articles
        )
    
    return {
        "total": total_articles,
        "by_month": dict(sorted(articles_by_month.items())),
        "by_year": dict(sorted(articles_by_year.items())),
        "by_language": dict(sorted(articles_by_language.items(), key=lambda x: x[1], reverse=True)),
        "by_source": dict(sorted(articles_by_source.items(), key=lambda x: x[1], reverse=True)),
        "by_source_month": {source: dict(months) for source, months in articles_by_source_month.items()}
    }

def process_single_jsonl(file_path, articles_by_month, articles_by_year, articles_by_language, articles_by_source, articles_by_source_month, total_articles):
    """Process a single JSONL file and update statistics."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                total_articles += 1
                
                # Extract publish date
                # Get date fields, handling None values
                date_publish = article.get('date_publish')
                date_download = article.get('date_download')
                date_modify = article.get('date_modify')
                
                relevant_dates = [date_download, date_modify] 
                # Filter out None values
                dates = [d for d in relevant_dates if d]
                
                # Get the latest date if any dates exist
                publish_date = parse_date(max(dates)) if dates else None
                if publish_date:
                    month_key = f"{publish_date.year}-{publish_date.month:02d}"
                    year_key = f"{publish_date.year}"
                    articles_by_month[month_key] += 1
                    articles_by_year[year_key] += 1
                
                # Extract language
                language = article.get('language', 'unknown')
                articles_by_language[language] += 1
                
                # Extract source domain
                source = article.get('source_domain', 'unknown')
                articles_by_source[source] += 1
                
                # Track source-month combination
                if publish_date and source != "unknown":
                    month_key = f"{publish_date.year}-{publish_date.month:02d}"
                    articles_by_source_month[source][month_key] += 1
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file_path}")
            except Exception as e:
                print(f"Error processing article: {e}")
    
    print(f"Processed {file_path}, total articles until now: {total_articles}")
    return total_articles

def print_stats(stats):
    """Print statistics in a readable format."""
    print("\n===== ARTICLE STATISTICS =====")
    print(f"Total articles: {stats['total']}")
    
    print("\n----- Articles by Year -----")
    for year, count in stats['by_year'].items():
        print(f"{year}: {count}")

    # print("\n----- Articles by Month -----")
    # for month, count in stats['by_month'].items():
    #     print(f"{month}: {count}")
    
    # print("\n----- Articles by Language -----")
    # for language, count in stats['by_language'].items():
    #     print(f"{language}: {count}")
    
    # print("\n----- Articles by Source Domain -----")
    # for source, count in stats['by_source'].items():
    #     print(f"{source}: {count}")

def plot_stats(stats, output_dir):
    """Generate plots for the statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot articles by year
    plt.figure(figsize=(10, 6))
    years = list(stats['by_year'].keys())
    year_counts = list(stats['by_year'].values())
    plt.bar(years, year_counts)
    plt.title('Articles by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(output_dir / 'articles_by_year.png')

    # Plot articles by month (from Jan 2016 to April 2025)
    plt.figure(figsize=(15, 8))
    
    # Create a list of all months from Jan 2016 to April 2025
    all_months = []
    # for year in range(2016, 2026):
    #     # For 2025, only include months up to April
    #     month_range = range(1, 13) if year < 2025 else range(1, 5)
    #     for month in month_range:
    #         all_months.append(f"{year}-{month:02d}")
    for year in range(2025, 2026):
        # For 2025, only include months up to April
        month_range = range(1, 13) 
        for month in month_range:
            all_months.append(f"{year}-{month:02d}")
    
    # Create a dictionary with all months initialized to 0
    monthly_counts = {month: 0 for month in all_months}
    
    # Update with actual counts from the data
    for month, count in stats['by_month'].items():
        year = int(month.split('-')[0])
        if 2016 <= year <= 2025:
            if month in monthly_counts:
                monthly_counts[month] = count
    
    months = list(monthly_counts.keys())
    counts = list(monthly_counts.values())
    
    plt.bar(months, counts)
    plt.xticks(rotation=90, fontsize=8)
    # plt.title('Articles by Month (Jan 2016 - Apr 2025)')
    plt.title('Articles by Month (2025)')
    plt.xlabel('Month')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(output_dir / 'articles_by_month.png')
    
    # Plot articles by language (all)
    plt.figure(figsize=(15, 8))
    languages = list(stats['by_language'].keys())
    counts = list(stats['by_language'].values())
    plt.bar(languages, counts)
    plt.xticks(rotation=90)
    plt.title('Articles by Language')
    plt.tight_layout()
    plt.savefig(output_dir / 'articles_by_language.png')
    
    # Plot articles by source (all)
    # For sources, we might have too many to display clearly
    # So we'll create a larger figure
    plt.figure(figsize=(20, 10))
    sources = list(stats['by_source'].keys())
    counts = list(stats['by_source'].values())
    
    # only keep sources with counts > 10
    sources = [source for source, count in stats['by_source'].items() if count > 1000]
    counts = [count for source, count in stats['by_source'].items() if count > 1000]
    
    # print these sources and counts to a file
    with open(output_dir / 'articles_by_source.txt', 'w') as f:
        for source, count in zip(sources, counts):
            f.write(f"{source}: {count}\n")
    
    plt.bar(sources, counts)
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Articles by Source Domain')
    plt.tight_layout()
    plt.savefig(output_dir / 'articles_by_source.png')
    
def print_source_by_month(stats):
    """
    First only keep the top 20 sources by total article count. 
    Then for each source, only PRINT the number of articles by month.
    """
    # Get top 20 sources by total article count
    top_sources = sorted(stats["by_source"].items(), key=lambda x: x[1], reverse=True)[:40]
    
    # Print the monthly counts for each source
    for source, total_count in top_sources:
        print(f"\nSource: {source} (Total: {total_count})")
        for month, count in stats["by_source_month"][source].items():
            print(f"  {month}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Process JSONL files and generate article statistics.')
    parser.add_argument('--input', help='Path to JSONL file or directory containing JSONL files', 
                        default='/fast/sgoel/forecasting/news/tokenized_data/news/deduped/')
    parser.add_argument('--output', '-o', default='plots/2025', help='Directory to save plots (default: plots)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist")
        return
    
    stats = process_jsonl_files(args.input)
    print_stats(stats)
    plot_stats(stats, args.output)
    print_source_by_month(stats)

if __name__ == "__main__":
    main()