import os
import json
import shutil
import argparse
import time
import math
import subprocess
import tempfile
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from pathlib import Path

# Default news directory if none is provided
DEFAULT_NEWS_DIR = "/is/cluster/fast/sgoel/forecasting/news/filtered_cc_articles/www.hindustantimes.com"

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

def analyze_file_batch(file_paths, batch_index, total_batches):
    """Analyze a batch of files and determine their destination folders (first stage)."""
    # Print progress for this batch
    print(f"Analyzing batch {batch_index+1}/{total_batches} ({len(file_paths)} files)")
    
    # Initialize mapping from file to its destination month
    file_destinations = {}
    monthly_counts = defaultdict(int)
    unknown_date_count = 0
    error_count = 0
    
    # For progress tracking within batch
    progress_interval = max(1, len(file_paths) // 10)  # Show progress ~10 times per batch
    
    for i, file_path in enumerate(file_paths):
        # Print progress within batch
        if (i + 1) % progress_interval == 0:
            print(f"  Analysis batch {batch_index+1}: Processed {i+1}/{len(file_paths)} files ({(i+1)/len(file_paths)*100:.1f}%)")
        
        try:
            # Parse the file to determine the month
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get dates
            date_modify = parse_date(data.get('date_modify'))
            date_publish = parse_date(data.get('date_publish'))
            
            # Determine month folder (prefer date_modify, fallback to date_publish)
            target_date = date_modify if date_modify else date_publish
            month_folder = get_month_year_key(target_date)
            
            if not month_folder:
                # If both dates are None/invalid
                unknown_date_count += 1
                continue
            
            # Store the destination folder for this file
            file_destinations[file_path] = month_folder
            monthly_counts[month_folder] += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Limit error output
                print(f"Error analyzing {file_path}: {e}")
    
    print(f"  Analysis batch {batch_index+1} completed: {len(file_destinations)} files categorized, {error_count} errors, {unknown_date_count} with unknown dates")
    return {
        'file_destinations': file_destinations,
        'monthly_counts': monthly_counts,
        'unknown_date_count': unknown_date_count,
        'error_count': error_count
    }

def move_files_using_external_commands(month_folder, file_list, target_dir, batch_index, total_batches):
    """Move files using external 'mv' command for better performance."""
    monthly_dir = os.path.join(target_dir, month_folder)
    
    # Create directory using shell command (mkdir -p)
    subprocess.run(["mkdir", "-p", monthly_dir], check=True)
    
    # For progress tracking
    total_files = len(file_list)
    progress_interval = max(1, total_files // 5)
    
    print(f"Starting move for month {month_folder}: {total_files} files (batch {batch_index+1}/{total_batches})")
    
    # Create a temporary file containing move commands
    moved_count = 0
    error_count = 0
    
    # Process in smaller chunks to optimize for system's command-line length limits
    chunk_size = 1000  # Adjust based on your system
    
    for chunk_start in range(0, len(file_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(file_list))
        current_chunk = file_list[chunk_start:chunk_end]
        
        # Create a temporary file with source paths to move
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            for file_path in current_chunk:
                temp_file.write(f"{file_path}\n")
            temp_file_name = temp_file.name
        
        try:
            # Use xargs to process the file list and mv command for moving files
            # --no-run-if-empty: don't run if no input
            # -d: delimiter is newline
            # -n: number of arguments per command
            process = subprocess.run(
                f"cat {temp_file_name} | xargs --no-run-if-empty -d '\n' -n 100 mv -t {monthly_dir}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode == 0:
                moved_count += len(current_chunk)
            else:
                # If bulk command fails, fall back to moving files individually
                print(f"Bulk move failed for {month_folder}, falling back to individual moves: {process.stderr}")
                individual_errors = 0
                for file_path in current_chunk:
                    try:
                        file_name = os.path.basename(file_path)
                        target_path = os.path.join(monthly_dir, file_name)
                        subprocess.run(["mv", file_path, target_path], check=True)
                        moved_count += 1
                    except Exception as e:
                        individual_errors += 1
                        error_count += 1
                        if individual_errors <= 5:  # Limit error output per chunk
                            print(f"Error moving {file_path}: {str(e)}")
                
        except Exception as e:
            error_count += len(current_chunk)
            print(f"Error processing chunk for {month_folder}: {str(e)}")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_name)
            except:
                pass
        
        # Print progress
        if chunk_end % progress_interval < chunk_size:
            print(f"  Month {month_folder} batch {batch_index+1}: Moved ~{moved_count}/{total_files} files ({moved_count/total_files*100:.1f}%)")
    
    print(f"  Month {month_folder} batch {batch_index+1} completed: moved {moved_count} files with {error_count} errors")
    return {
        'month_folder': month_folder,
        'moved_count': moved_count,
        'error_count': error_count
    }

def create_all_month_directories(months, target_dir):
    """Create all month directories at once using external commands."""
    if not months:
        return
    
    # Create a temporary file with the directories to create
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        for month in months:
            monthly_dir = os.path.join(target_dir, month)
            temp_file.write(f"{monthly_dir}\n")
        temp_file_name = temp_file.name
    
    try:
        # Use xargs to create all directories at once
        subprocess.run(
            f"cat {temp_file_name} | xargs --no-run-if-empty -d '\n' mkdir -p",
            shell=True,
            check=True
        )
    except Exception as e:
        print(f"Error creating month directories: {str(e)}")
        # Fall back to creating directories one by one
        for month in months:
            monthly_dir = os.path.join(target_dir, month)
            try:
                os.makedirs(monthly_dir, exist_ok=True)
            except Exception:
                pass
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file_name)
        except:
            pass

def reorganize_files_by_month_parallel(news_dir=None, num_processes=None):
    """Reorganize files into monthly folders using external commands for better performance."""
    # Use provided news_dir or fall back to default
    target_dir = news_dir if news_dir else DEFAULT_NEWS_DIR
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)  # Use at most 16 processes
    
    print(f"Starting to reorganize files in: {target_dir}")
    print(f"Using {num_processes} parallel processes with external commands")
    
    start_time = time.time()
    
    # Scan all files first
    all_files = []
    print("Scanning directory for files...")
    
    # Use find command for faster directory scanning
    try:
        process = subprocess.run(
            f"find {target_dir} -maxdepth 1 -type f -name '*.json'",
            shell=True,
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )
        all_files = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        print(f"Found {len(all_files)} files using 'find' command")
    except Exception as e:
        print(f"External file scan failed: {str(e)}. Falling back to Python scan...")
        
        # Fall back to Python scan if external command fails
        scan_count = 0
        progress_interval = 10000
        
        with os.scandir(target_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    all_files.append(entry.path)
                    scan_count += 1
                    if scan_count % progress_interval == 0:
                        print(f"  Scanned {scan_count} files so far...")
    
    total_files = len(all_files)
    print(f"Found {total_files} files to process")
    
    if total_files == 0:
        print("No files to process. Exiting.")
        return
    
    # STAGE 1: Analysis - Split files into batches for parallel analysis
    print("\n===== STAGE 1: ANALYZING FILES =====")
    batch_size = math.ceil(total_files / num_processes)
    file_batches = [all_files[i:i+batch_size] for i in range(0, total_files, batch_size)]
    
    # Analyze files in parallel to determine their month folders
    with mp.Pool(processes=num_processes) as pool:
        args_list = [(batch, i, len(file_batches)) for i, batch in enumerate(file_batches)]
        analysis_results = pool.starmap(analyze_file_batch, args_list)
    
    # Combine analysis results
    file_destinations = {}
    monthly_counts = defaultdict(int)
    unknown_date_count = 0
    error_count = 0
    
    for result in analysis_results:
        file_destinations.update(result['file_destinations'])
        for month, count in result['monthly_counts'].items():
            monthly_counts[month] += count
        unknown_date_count += result['unknown_date_count']
        error_count += result['error_count']
    
    print(f"\nAnalysis complete. {len(file_destinations)} files categorized into {len(monthly_counts)} month folders.")
    print(f"Files with unknown dates: {unknown_date_count}")
    print(f"Analysis errors: {error_count}")
    
    # STAGE 2: Move files - Create a task list organized by months
    print("\n===== STAGE 2: MOVING FILES =====")
    
    # Create all month directories upfront using a single external command
    print("Creating all month directories...")
    create_all_month_directories(monthly_counts.keys(), target_dir)
    
    # Organize files by month folder
    files_by_month = defaultdict(list)
    for file_path, month_folder in file_destinations.items():
        files_by_month[month_folder].append(file_path)
    
    # Prepare tasks for parallel processing
    move_tasks = []
    
    # Sort months by size and process smaller months first
    months_by_size = sorted(files_by_month.items(), key=lambda x: len(x[1]))
    
    for month_folder, file_list in months_by_size:
        # Split large months into multiple batches
        if len(file_list) > 10000:  # Adjust based on performance testing
            split_batch_size = 5000  # Larger batch size for external commands
            for i in range(0, len(file_list), split_batch_size):
                move_tasks.append((month_folder, file_list[i:i+split_batch_size]))
        else:
            move_tasks.append((month_folder, file_list))
    
    print(f"Moving files in {len(move_tasks)} batches using external commands...")
    
    # Move files in parallel using external commands
    with mp.Pool(processes=num_processes) as pool:
        args_list = [
            (month_folder, file_list, target_dir, i, len(move_tasks)) 
            for i, (month_folder, file_list) in enumerate(move_tasks)
        ]
        move_results = pool.starmap(move_files_using_external_commands, args_list)
    
    # Combine move results
    moved_count = sum(result['moved_count'] for result in move_results)
    move_error_count = sum(result['error_count'] for result in move_results)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n===== FILE REORGANIZATION SUMMARY =====")
    print(f"Total files processed: {total_files}")
    print(f"Files successfully moved: {moved_count}")
    print(f"Files with analysis errors: {error_count}")
    print(f"Files with move errors: {move_error_count}")
    print(f"Files with unknown dates: {unknown_date_count}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    if elapsed_time > 0:
        print(f"Average processing speed: {total_files/elapsed_time:.2f} files/second")
    
    print("\nMonthly distribution:")
    for month, count in sorted(monthly_counts.items()):
        print(f"  {month}: {count} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize news JSON files into monthly folders")
    parser.add_argument('news_dir', nargs='?', type=str, default=None,
                        help="Directory containing the news JSON files (optional)")
    parser.add_argument('--processes', type=int, default=None,
                        help="Number of parallel processes to use (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    reorganize_files_by_month_parallel(args.news_dir, args.processes)