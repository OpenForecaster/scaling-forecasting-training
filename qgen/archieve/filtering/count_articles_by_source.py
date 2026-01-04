#!/usr/bin/env python3
import os
import json
import glob
from collections import defaultdict
import re
from datetime import datetime

def extract_news_source_from_filename(filename):
    """Extract news source from filename pattern like 'deepseek-chat-v3-0324_forbes-2023_50000_free_3.jsonl'"""
    # Pattern: deepseek-chat-v3-0324_SOURCE-YEAR_SIZE_free_3.jsonl
    match = re.match(r'.*?_(.+?)-(\d{4})(?:-\d{2})?_\d+_free_3\.jsonl', filename)
    if match:
        source = match.group(1)
        return source
    return "unknown"

def extract_year_from_date_download(date_str):
    """Extract year from date_download field like '2023-12-10 15:36:44+00:00'"""
    if not date_str:
        return "unknown"
    try:
        # Parse the date string and extract year
        date_obj = datetime.strptime(date_str.split('+')[0], '%Y-%m-%d %H:%M:%S')
        return str(date_obj.year)
    except (ValueError, AttributeError):
        return "unknown"

def count_articles_in_file(file_path):
    """Count the number of articles (lines) and track by year in a JSONL file"""
    try:
        year_counts = defaultdict(int)
        line_count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    date_download = data.get('date_download')
                    year = extract_year_from_date_download(date_download)
                    year_counts[year] += 1
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
        
        return line_count, year_counts
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, defaultdict(int)

def main():
    directory = "/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/qgen/clean"
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return
    
    # Get all JSONL files
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in directory")
        return
    
    # Count articles by source and year
    source_counts = defaultdict(int)
    year_counts = defaultdict(int)
    total_articles = 0
    
    print("Analyzing news sources and article counts...")
    print("=" * 60)
    
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        source = extract_news_source_from_filename(filename)
        article_count, file_year_counts = count_articles_in_file(file_path)
        
        source_counts[source] += article_count
        total_articles += article_count
        
        # Add year counts from this file
        for year, count in file_year_counts.items():
            year_counts[year] += count
        
        print(f"{filename}")
        print(f"  Source: {source}")
        print(f"  Articles: {article_count:,}")
        print()
    
    print("=" * 60)
    print("SUMMARY BY NEWS SOURCE:")
    print("=" * 60)
    
    # Sort by article count (descending)
    sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
    
    for source, count in sorted_sources:
        percentage = (count / total_articles) * 100
        print(f"{source:20} {count:8,} articles ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("SUMMARY BY YEAR:")
    print("=" * 60)
    
    # Sort by year (ascending)
    sorted_years = sorted(year_counts.items(), key=lambda x: x[0] if x[0] != "unknown" else "9999")
    
    for year, count in sorted_years:
        if year != "unknown":
            percentage = (count / total_articles) * 100
            print(f"{year:20} {count:8,} articles ({percentage:5.1f}%)")
        else:
            print(f"{year:20} {count:8,} articles (unknown year)")
    
    print("=" * 60)
    print(f"TOTAL ARTICLES: {total_articles:,}")
    print(f"TOTAL SOURCES: {len(source_counts)}")
    print(f"TOTAL YEARS: {len([y for y in year_counts.keys() if y != 'unknown'])}")

if __name__ == "__main__":
    main() 