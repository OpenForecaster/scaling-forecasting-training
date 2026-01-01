#!/usr/bin/env python3
"""
Script to combine two JSONL files, deduplicate by URL, and sort by date.

This script:
1. Loads entries from two JSONL files (primary and secondary)
2. Sorts them by max(date_modify, date_download)
3. Keeps only unique entries based on URL (latest entry wins)
4. Saves the final combined file with _combined suffix

Usage:
    python keep_unique.py --primary /path/to/primary.jsonl --secondary /path/to/secondary.jsonl
"""

import json
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object."""
    if not date_str:
        return None
    
    # Handle different date formats
    formats = [
        "%Y-%m-%d %H:%M:%S%z",  # 2022-10-14 18:25:58+00:00
        "%Y-%m-%d %H:%M:%S",     # 2022-10-14 00:00:00
        "%Y-%m-%dT%H:%M:%S%z",   # 2022-10-14T18:25:58+00:00
        "%Y-%m-%dT%H:%M:%SZ",    # 2022-10-14T18:25:58Z
        "%Y-%m-%d"               # 2022-10-14
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, try to handle timezone info manually
    try:
        # Remove timezone info if present and try again
        cleaned_date = date_str.replace('+00:00', '').replace('Z', '')
        return datetime.fromisoformat(cleaned_date)
    except:
        pass
    
    return None

def get_latest_date(article: Dict[str, Any]) -> Optional[datetime]:
    """Get the latest date from date_modify or date_download."""
    date_modify = article.get('date_modify')
    date_download = article.get('date_download')
    
    dates = []
    if date_modify:
        parsed_modify = parse_date(date_modify)
        if parsed_modify:
            dates.append(parsed_modify)
    
    if date_download:
        parsed_download = parse_date(date_download)
        if parsed_download:
            dates.append(parsed_download)
    
    return max(dates) if dates else None

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load articles from a JSONL file."""
    articles = []
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return articles
    
    print(f"Loading articles from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line.strip())
                if article:  # Skip empty objects
                    articles.append(article)
        
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing JSON at line {line_num} in {file_path}: {e}")
                continue
    
    print(f"Loaded {len(articles)} articles from {file_path}")
    return articles

def combine_and_deduplicate(primary_articles: List[Dict[str, Any]], 
                          secondary_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine articles from both sources and deduplicate by URL, keeping the latest version."""
    
    # Dictionary to store the latest version of each article by URL
    url_to_article = {}
    exact_url = "https://www.dw.com/en/bangladesh-trains-canceled-as-staff-strike-over-benefits/a-71431255?maca=en-rss-en-top-1022-xml-atom"
    # Process all articles
    all_articles = primary_articles + secondary_articles
    # all_articles = secondary_articles
    print(f"Processing {len(all_articles)} total articles...")
    
    for article in all_articles:
        url = article.get('filename').lower()
        
        # if "bangladesh" in url and "trains" in url:
        #     print("Found bangladesh url:\n" + url)
        #     # print(article)
        #     print(exact_url)
        #     # exit()
        
        # if url == exact_url.lower():
        #     print("Found exact url: ", url)
        #     print(article)
        #     exit()
        
        if not url:
            print(f"Warning: Article without URL found, skipping: {article.get('title', 'No title')[:50]}...")
            continue
        
        latest_date = get_latest_date(article)
        
        # If we haven't seen this URL before, or if this article is newer
        if url not in url_to_article:
            url_to_article[url] = (article, latest_date)
        else:
            print("Existing article found: ", url)
            existing_article, existing_date = url_to_article[url]
            
            # Compare dates (handle None dates)
            if latest_date is None and existing_date is None:
                # Both have no date, keep the first one
                continue
            elif latest_date is None:
                # Current has no date, keep existing
                continue
            elif existing_date is None:
                # Existing has no date, use current
                url_to_article[url] = (article, latest_date)
            elif latest_date > existing_date:
                # Current is newer
                url_to_article[url] = (article, latest_date)
            # Otherwise keep existing (it's newer or same)
    
    # Extract just the articles and sort by date
    unique_articles = [article for article, date in url_to_article.values()]
    
    # Sort articles by date (newest first)
    def sort_key(article):
        date = get_latest_date(article)
        return date if date else datetime.min
    
    unique_articles.sort(key=sort_key, reverse=True)
    
    print(f"After deduplication: {len(unique_articles)} unique articles")
    return unique_articles

def save_combined_file(articles: List[Dict[str, Any]], primary_file: str) -> str:
    """Save the combined articles to a new file with _combined suffix."""
    
    # Generate output filename
    primary_path = Path(primary_file)
    output_file = primary_path.parent / f"{primary_path.stem}_combined{primary_path.suffix}"
    
    print(f"Saving {len(articles)} articles to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully saved combined file: {output_file}")
    return str(output_file)

def print_statistics(primary_articles: List[Dict[str, Any]], 
                    secondary_articles: List[Dict[str, Any]], 
                    combined_articles: List[Dict[str, Any]]):
    """Print statistics about the combination process."""
    print("\n===== COMBINATION STATISTICS =====")
    print(f"Primary file articles:   {len(primary_articles)}")
    print(f"Secondary file articles: {len(secondary_articles)}")
    print(f"Total articles:          {len(primary_articles) + len(secondary_articles)}")
    print(f"Unique articles:         {len(combined_articles)}")
    print(f"Duplicates removed:      {len(primary_articles) + len(secondary_articles) - len(combined_articles)}")
    
    # Show date range of combined articles
    dates = [get_latest_date(article) for article in combined_articles]
    valid_dates = [d for d in dates if d is not None]
    
    if valid_dates:
        print(f"Date range: {min(valid_dates).strftime('%Y-%m-%d')} to {max(valid_dates).strftime('%Y-%m-%d')}")
        print(f"Articles with valid dates: {len(valid_dates)}/{len(combined_articles)}")

def main():
    parser = argparse.ArgumentParser(description="Combine and deduplicate JSONL files by URL")
    parser.add_argument("--primary", type=str, required=True,
                       help="Path to primary JSONL file")
    parser.add_argument("--secondary", type=str, required=True,
                       help="Path to secondary JSONL file")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.primary):
        print(f"Error: Primary file does not exist: {args.primary}")
        return 1
    
    if not os.path.exists(args.secondary):
        print(f"Error: Secondary file does not exist: {args.secondary}")
        return 1
    
    print(f"Primary file: {args.primary}")
    print(f"Secondary file: {args.secondary}")
    
    # Load articles from both files
    primary_articles = load_jsonl_file(args.primary)
    secondary_articles = load_jsonl_file(args.secondary)
    
    if not primary_articles and not secondary_articles:
        print("Error: No articles found in either file")
        return 1
    
    # Combine and deduplicate
    combined_articles = combine_and_deduplicate(primary_articles, secondary_articles)
    
    if not combined_articles:
        print("Error: No articles remaining after processing")
        return 1
    
    # Save combined file
    output_file = save_combined_file(combined_articles, args.primary)
    
    # Print statistics
    print_statistics(primary_articles, secondary_articles, combined_articles)
    
    print(f"\n✅ Successfully created combined file: {output_file}")
    return 0

if __name__ == "__main__":
    exit(main())
