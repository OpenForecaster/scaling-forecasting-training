#!/usr/bin/env python3
import os
import subprocess
from collections import defaultdict
import re
from pathlib import Path
import tqdm

def count_articles_by_month():
    # Path to the filtered articles directory
    base_dir = "/fast/sgoel/forecasting/news/filtered_cc_articles_2025"
    # base_dir = "/fast/sgoel/forecasting/news/articles2025/deduped"
    
    # Dictionary to store counts by year-month
    monthly_counts = defaultdict(int)
    
    # Get all domain directories
    domain_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for domain in tqdm.tqdm(domain_dirs):
        domain_path = os.path.join(base_dir, domain)
        sum_domain_articles = 0 
        
        # Use find command to get all directories that match year-month pattern
        cmd = f"find {domain_path} -type d -name '[0-9][0-9][0-9][0-9]-[0-9][0-9]' -print"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        month_dirs = result.stdout.strip().split('\n')
        
        # Filter out empty entries
        month_dirs = [d for d in month_dirs if d]
        
        # Count files in each month directory using fast bash commands
        for month_dir in month_dirs:
            if not os.path.exists(month_dir):
                continue
                
            # Extract the year-month from the path
            year_month = os.path.basename(month_dir)
            
            # Use find with wc to count files quickly   
            cmd = f"find {month_dir} -type f -name '*.json' | wc -l"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            count = int(result.stdout.strip())
            sum_domain_articles += count
            monthly_counts[year_month] += count
        
        # Also count any JSON files directly in the domain directory
        cmd = f"find {domain_path} -maxdepth 1 -type f -name '*.json' | wc -l"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        count = int(result.stdout.strip())
        sum_domain_articles += count
        if count > 0:
            monthly_counts["root_dir"] += count
        print(f"Domain {domain} has {sum_domain_articles} articles, {count} in root dir")

    # Sort by year and month
    sorted_months = sorted(
        [m for m in monthly_counts.keys() if m != "root_dir"], 
        key=lambda x: x if x != "root_dir" else "0000-00"
    )
    
    # Print results
    print(f"{'Month':<10} | {'Count':>10}")
    print("-" * 23)
    
    for month in sorted_months:
        print(f"{month:<10} | {monthly_counts[month]:>10,}")
    
    if "root_dir" in monthly_counts and monthly_counts["root_dir"] > 0:
        print(f"{'root_dir':<10} | {monthly_counts['root_dir']:>10,}")
    
    total = sum(monthly_counts.values())
    print("-" * 23)
    print(f"{'TOTAL':<10} | {total:>10,}")

if __name__ == "__main__":
    count_articles_by_month()