import os
import json
import shutil
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import math
from functools import partial

NEWS_DIR = "/is/cluster/fast/sgoel/forecasting/news/filtered_cc_articles/www.hindustantimes.com"

def parse_date(date_str):
    """Parse date string to datetime object."""
    if not date_str or date_str == 'None':
        return None
    
    try:
        # Handle timezone in date_download
        if '+' in date_str:
            # Strip timezone for simplicity
            date_str = date_str.split('+')[0]
        
        # Parse the date
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        try:
            # Try alternative format if needed
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
        except Exception:
            return None

def get_month_year_key(dt):
    """Get month-year key from datetime object."""
    if dt is None:
        return None
    return f"{dt.year}-{dt.month:02d}"

def process_file_batch(file_paths, batch_index, total_batches, batch_size):
    """Process a batch of files and return statistics."""
    # Print progress for this batch
    print(f"Starting batch {batch_index+1}/{total_batches} ({len(file_paths)} files)")
    
    # Statistics containers
    publish_monthly_counts = defaultdict(int)
    
    # For calculating differences
    download_publish_diffs = []  # Time differences in hours
    modify_publish_diffs = []    # Time differences in hours
    
    # For date_modify stats
    total_files = 0
    modify_none_count = 0
    
    # For progress tracking
    progress_interval = max(1, len(file_paths) // 10)  # Show progress ~10 times per batch
    
    for i, file_path in enumerate(file_paths):
        # Print progress within batch
        if (i + 1) % progress_interval == 0:
            print(f"  Batch {batch_index+1}: Processed {i+1}/{len(file_paths)} files ({(i+1)/len(file_paths)*100:.1f}%)")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract dates
            date_download_str = data.get('date_download')
            date_modify_str = data.get('date_modify')
            date_publish_str = data.get('date_publish')
            
            # Parse dates
            date_download = parse_date(date_download_str)
            date_modify = parse_date(date_modify_str)
            date_publish = parse_date(date_publish_str)
            
            # Count by month-year for publish date
            if date_publish:
                publish_key = get_month_year_key(date_publish)
                if publish_key:
                    publish_monthly_counts[publish_key] += 1
            
            # Calculate time difference between download and publish
            if date_download and date_publish:
                diff_hours = (date_download - date_publish).total_seconds() / 3600
                download_publish_diffs.append(diff_hours)
            
            # Calculate time difference between modify and publish
            if date_modify and date_publish:
                diff_hours = (date_modify - date_publish).total_seconds() / 3600
                modify_publish_diffs.append(diff_hours)
            
            # Track modify None count
            total_files += 1
            if date_modify_str is None or date_modify is None:
                modify_none_count += 1
                
        except Exception as e:
            # Skip problematic files
            continue
    
    print(f"  Batch {batch_index+1} completed")
    return {
        'publish_monthly_counts': publish_monthly_counts,
        'download_publish_diffs': download_publish_diffs,
        'modify_publish_diffs': modify_publish_diffs,
        'total_files': total_files,
        'modify_none_count': modify_none_count
    }

def process_files_parallel(num_processes=8):
    """Process all files in parallel and collect statistics."""
    # Get all file paths
    all_files = []
    try:
        print("Scanning directory for files...")
        file_count = 0
        progress_interval = 10000  # Print progress every 10,000 files
        with os.scandir(NEWS_DIR) as entries:
            for entry in entries:
                if entry.is_file():
                    all_files.append(entry.path)
                    file_count += 1
                    if file_count % progress_interval == 0:
                        print(f"  Scanned {file_count} files so far...")
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return
    
    total_files = len(all_files)
    print(f"Found {total_files} files to process")
    
    # Split files into batches for parallel processing
    batch_size = math.ceil(total_files / num_processes)
    file_batches = [all_files[i:i+batch_size] for i in range(0, total_files, batch_size)]
    
    # Process batches in parallel
    print(f"Processing files using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        # Use partial to pass additional parameters to process_file_batch
        process_func = partial(
            process_file_batch, 
            total_batches=len(file_batches), 
            batch_size=batch_size
        )
        
        # Process batches with their index for progress reporting
        results = pool.starmap(
            process_func, 
            [(batch, i, len(file_batches)) for i, batch in enumerate(file_batches)]
        )
    
    # Combine results
    publish_monthly_counts = defaultdict(int)
    download_publish_diffs = []
    modify_publish_diffs = []
    total_processed = 0
    modify_none_count = 0
    
    for result in results:
        # Combine monthly counts
        for month_year, count in result['publish_monthly_counts'].items():
            publish_monthly_counts[month_year] += count
        
        # Combine time differences
        download_publish_diffs.extend(result['download_publish_diffs'])
        modify_publish_diffs.extend(result['modify_publish_diffs'])
        
        # Combine file counts
        total_processed += result['total_files']
        modify_none_count += result['modify_none_count']
    
    # Calculate statistics
    modify_none_fraction = modify_none_count / total_processed if total_processed > 0 else 0
    
    # Calculate percentiles for download-publish difference
    download_publish_stats = {}
    if download_publish_diffs:
        download_publish_stats = {
            'mean': np.mean(download_publish_diffs),
            'median': np.median(download_publish_diffs),
            'p25': np.percentile(download_publish_diffs, 25),
            'p75': np.percentile(download_publish_diffs, 75),
            'min': min(download_publish_diffs),
            'max': max(download_publish_diffs)
        }
    
    # Calculate percentiles for modify-publish difference
    modify_publish_stats = {}
    if modify_publish_diffs:
        modify_publish_stats = {
            'mean': np.mean(modify_publish_diffs),
            'median': np.median(modify_publish_diffs),
            'p25': np.percentile(modify_publish_diffs, 25),
            'p75': np.percentile(modify_publish_diffs, 75),
            'min': min(modify_publish_diffs),
            'max': max(modify_publish_diffs)
        }
    
    # Print results
    print(f"\nTotal files processed: {total_processed}")
    
    print("\n===== MONTHLY FREQUENCY OF DATE_PUBLISH =====")
    for month_year in sorted(publish_monthly_counts.keys()):
        print(f"{month_year}: {publish_monthly_counts[month_year]} articles")
    
    print("\n===== DOWNLOAD-PUBLISH TIME DIFFERENCE STATISTICS (HOURS) =====")
    if download_publish_stats:
        print(f"Mean: {download_publish_stats['mean']:.2f} hours")
        print(f"Median: {download_publish_stats['median']:.2f} hours")
        print(f"25th percentile: {download_publish_stats['p25']:.2f} hours")
        print(f"75th percentile: {download_publish_stats['p75']:.2f} hours")
        print(f"Min: {download_publish_stats['min']:.2f} hours")
        print(f"Max: {download_publish_stats['max']:.2f} hours")
    else:
        print("No valid download-publish time differences found")
    
    print("\n===== MODIFY-PUBLISH TIME DIFFERENCE STATISTICS (HOURS) =====")
    if modify_publish_stats:
        print(f"Mean: {modify_publish_stats['mean']:.2f} hours")
        print(f"Median: {modify_publish_stats['median']:.2f} hours")
        print(f"25th percentile: {modify_publish_stats['p25']:.2f} hours")
        print(f"75th percentile: {modify_publish_stats['p75']:.2f} hours")
        print(f"Min: {modify_publish_stats['min']:.2f} hours")
        print(f"Max: {modify_publish_stats['max']:.2f} hours")
    else:
        print("No valid modify-publish time differences found")
    
    print("\n===== DATE_MODIFY STATISTICS =====")
    print(f"Fraction of articles with date_modify = None: {modify_none_fraction:.4f} ({modify_none_count}/{total_processed})")

if __name__ == "__main__":
    process_files_parallel(8)  # Use 8 processes for analysis