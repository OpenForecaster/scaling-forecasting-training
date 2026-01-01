import lmdb
import json
import os
import pickle
import hashlib
import random
import time
from tqdm import tqdm
import concurrent.futures
import threading
import queue
import argparse
import sys
import subprocess

def log_message(log_file, message):
    """Log a message to both console and log file"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def process_directory(directory, output_dir, shard_paths, shards, verify_sample, delete_jsons, log_file):
    """Process a single directory of JSON files"""
    # Get all JSON files in this directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        return 0, 0  # No files to process
    
    dir_processed = 0
    dir_verified = 0
    
    try:
        # Open all shard environments for this thread
        envs = [lmdb.open(path, map_size=int(1e11)) for path in shard_paths]
        
        # Create main databases
        dbs = []
        for env in envs:
            articles_db = env.open_db(b'articles')
            metadata_db = env.open_db(b'metadata')
            dbs.append((articles_db, metadata_db))
        
        # Process each JSON file
        for file in json_files:
            try:
                file_path = os.path.join(directory, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                # Extract document ID from filename
                doc_id = os.path.splitext(file)[0]
                
                # Determine which shard to use based on hash of doc_id
                shard_idx = int(hashlib.md5(doc_id.encode('utf-8')).hexdigest(), 16) % shards
                
                # Get the environment, databases for this shard
                env = envs[shard_idx]
                articles_db, metadata_db = dbs[shard_idx]
                
                # Prepare article data
                article_json = json.dumps(article).encode('utf-8')
                
                # Extract metadata
                metadata = {
                    'id': doc_id,
                    'title': article.get('title', ''),
                    'authors': article.get('authors', []),
                    'date_download': article.get('date_download', None),
                    'date_modify': article.get('date_modify', None),
                    'date_publish': article.get('date_publish', None),
                    'description': article.get('description', ''),
                    'filename': article.get('filename', ''),
                    'image_url': article.get('image_url', ''),
                    'language': article.get('language', ''),
                    'localpath': article.get('localpath', ''),
                    'maintext': article.get('maintext', ''),
                    'source_domain': article.get('source_domain', ''),
                    'title_page': article.get('title_page', ''),
                    'title_rss': article.get('title_rss', ''),
                    'url': article.get('url', '')
                }
                metadata_pickle = pickle.dumps(metadata)
                
                # Store in LMDB
                with env.begin(write=True) as txn:
                    # Store article
                    key = doc_id.encode('utf-8')
                    txn.put(key, article_json, db=articles_db)
                    
                    # Store metadata
                    txn.put(key, metadata_pickle, db=metadata_db)
                
                dir_processed += 1
                
                # Verify a sample of documents
                if random.random() < verify_sample:
                    with env.begin() as txn:
                        # Verify article
                        stored_article = txn.get(key, db=articles_db)
                        if stored_article:
                            stored_article_json = json.loads(stored_article.decode('utf-8'))
                            if stored_article_json == article:
                                dir_verified += 1
                            else:
                                log_message(log_file, f"Verification failed for {file_path}: content mismatch")
                        else:
                            log_message(log_file, f"Verification failed for {file_path}: not found")
                
                # Delete JSON file if requested and successfully stored
                if delete_jsons:
                    os.remove(file_path)
            
            except Exception as e:
                log_message(log_file, f"Error processing {file_path}: {str(e)}")
        
        # Close environments
        for env in envs:
            env.close()
        
        return dir_processed, dir_verified
    
    except Exception as e:
        log_message(log_file, f"Error processing directory {directory}: {str(e)}")
        return 0, 0

def parallel_convert_directories(json_dir, output_dir, shards=20, max_workers=48, 
                               verify_sample=0.01, delete_jsons=False):
    """Convert JSON files to LMDB in parallel"""
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create shard directories
    shard_paths = []
    for i in range(shards):
        shard_path = os.path.join(output_dir, f'shard_{i}')
        os.makedirs(shard_path, exist_ok=True)
        shard_paths.append(shard_path)
    
    # Create log file
    log_file = os.path.join(output_dir, "conversion_log.txt")
    log_message(log_file, f"Starting LMDB conversion from {json_dir} to {output_dir}")
    log_message(log_file, f"Shards: {shards}, Workers: {max_workers}, Verify: {verify_sample*100}%, Delete JSONs: {delete_jsons}")
    
    # Get all directories containing JSON files
    all_dirs = []
    # Use subprocess to run 'find' command which is much faster for large directory structures
    try:
        # Find all directories under json_dir
        cmd = ["find", json_dir, "-type", "d"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        all_dirs = result.stdout.strip().split('\n')
    except subprocess.SubprocessError as e:
        log_message(log_file, f"Error using find command: {str(e)}")
        # Fallback to a simple list with just the root directory
        all_dirs = [json_dir]
    
    total_dirs = len(all_dirs)
    log_message(log_file, f"Found {total_dirs} directories containing JSON files")
    
    # Create a file to track processed directories
    processed_dirs_path = os.path.join(output_dir, "processed_dirs.txt")
    
    # If resuming, load already processed directories
    already_processed = set()
    if os.path.exists(processed_dirs_path):
        with open(processed_dirs_path, 'r') as f:
            already_processed = set(line.strip() for line in f)
        
        log_message(log_file, f"Resuming conversion. {len(already_processed)} directories already processed.")
    
    # Filter out already processed directories
    dirs_to_process = [d for d in all_dirs if d not in already_processed]
    
    # Process directories in parallel
    log_message(log_file, f"Starting parallel conversion with {max_workers} workers...")
    
    total_processed = 0
    total_verified = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all directory processing tasks
        future_to_dir = {
            executor.submit(
                process_directory, 
                directory, 
                output_dir, 
                shard_paths, 
                shards, 
                verify_sample, 
                delete_jsons,
                log_file
            ): directory for directory in dirs_to_process
        }
        
        # Process completed tasks and update processed directories file
        with open(processed_dirs_path, 'a') as processed_file:
            for future in concurrent.futures.as_completed(future_to_dir):
                directory = future_to_dir[future]
                try:
                    processed, verified = future.result()
                    total_processed += processed
                    total_verified += verified
                    # Mark directory as processed
                    processed_file.write(directory + '\n')
                    processed_file.flush()
                    log_message(log_file, f"Completed {directory}, processed {processed} documents, verified {verified}. Total: {total_processed}")
                except Exception as e:
                    log_message(log_file, f"ERROR in directory {directory}: {str(e)}")
    
    # Calculate statistics
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    log_message(log_file, f"Conversion complete. Total documents processed: {total_processed}")
    if total_processed > 0:
        log_message(log_file, f"Verification rate: {total_verified/total_processed:.2%}")
    log_message(log_file, f"Elapsed time: {elapsed_time:.2f} seconds")
    log_message(log_file, f"Processing rate: {total_processed/elapsed_time:.2f} documents/second")
    
    return total_processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to LMDB database")
    
    parser.add_argument('json_dir', type=str, 
                       help="Directory containing the JSON files to convert")
    
    parser.add_argument('output_dir', type=str,
                       help="Directory to store the LMDB database")
    
    parser.add_argument('--shards', type=int, default=20,
                       help="Number of LMDB shards to create (default: 20)")
    
    parser.add_argument('--workers', type=int, default=48,
                       help="Number of parallel workers to use (default: 48)")
    
    parser.add_argument('--verify', type=float, default=0.01,
                       help="Fraction of documents to verify (default: 0.01 = 1%%)")
    
    parser.add_argument('--delete', action='store_true',
                       help="Delete JSON files after successful conversion")
    
    args = parser.parse_args()
    
    # Convert JSON files to LMDB
    parallel_convert_directories(
        args.json_dir,
        args.output_dir,
        shards=args.shards,
        max_workers=args.workers,
        verify_sample=args.verify,
        delete_jsons=args.delete
    )