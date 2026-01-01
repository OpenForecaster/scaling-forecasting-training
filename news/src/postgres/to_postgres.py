import json
import os
import hashlib
import random
import time
import psycopg2
from psycopg2.extras import execute_batch
import concurrent.futures
import argparse
import subprocess
from tqdm import tqdm

def log_message(log_file, message):
    """Log a message to both console and log file"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def get_db_connection(db_name, user, password, host, port):
    """Create a database connection"""
    return psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )

def setup_database(conn):
    """Set up the database schema"""
    with conn.cursor() as cur:
        # Create articles table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            data JSONB NOT NULL
        )
        """)
        
        # Create metadata table with indexes for common query fields
        cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT[],
            date_download TIMESTAMP,
            date_modify TIMESTAMP,
            date_publish TIMESTAMP,
            description TEXT,
            filename TEXT,
            image_url TEXT,
            language TEXT,
            localpath TEXT,
            maintext TEXT,
            source_domain TEXT,
            title_page TEXT,
            title_rss TEXT,
            url TEXT
        )
        """)
        
        # Create indexes for common query fields
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metadata_date_publish ON metadata(date_publish)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metadata_source_domain ON metadata(source_domain)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metadata_language ON metadata(language)")
        
        conn.commit()

def process_directory(directory, db_params, verify_sample, delete_jsons, log_file, batch_size=1000):
    """Process a single directory of JSON files"""
    # Get all JSON files in this directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        return 0, 0  # No files to process
    
    dir_processed = 0
    dir_verified = 0
    
    try:
        # Connect to the database
        conn = get_db_connection(**db_params)
        
        # Process files in batches
        articles_batch = []
        metadata_batch = []
        
        for file in json_files:
            try:
                file_path = os.path.join(directory, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                # Extract document ID from filename
                doc_id = os.path.splitext(file)[0]
                
                # Prepare article data
                articles_batch.append((doc_id, json.dumps(article)))
                
                # Extract metadata
                metadata = (
                    doc_id,
                    article.get('title', ''),
                    article.get('authors', []),
                    article.get('date_download'),
                    article.get('date_modify'),
                    article.get('date_publish'),
                    article.get('description', ''),
                    article.get('filename', ''),
                    article.get('image_url', ''),
                    article.get('language', ''),
                    article.get('localpath', ''),
                    article.get('maintext', ''),
                    article.get('source_domain', ''),
                    article.get('title_page', ''),
                    article.get('title_rss', ''),
                    article.get('url', '')
                )
                metadata_batch.append(metadata)
                
                # If we've reached the batch size, insert the batch
                if len(articles_batch) >= batch_size:
                    insert_batch(conn, articles_batch, metadata_batch)
                    dir_processed += len(articles_batch)
                    
                    # Verify a sample if needed
                    if verify_sample > 0:
                        verified = verify_batch(conn, articles_batch, verify_sample, log_file)
                        dir_verified += verified
                    
                    # Delete JSON files if requested
                    if delete_jsons:
                        for i, (doc_id, _) in enumerate(articles_batch):
                            file_to_delete = os.path.join(directory, f"{doc_id}.json")
                            if os.path.exists(file_to_delete):
                                os.remove(file_to_delete)
                    
                    # Clear batches
                    articles_batch = []
                    metadata_batch = []
            
            except Exception as e:
                log_message(log_file, f"Error processing {file_path}: {str(e)}")
        
        # Insert any remaining items
        if articles_batch:
            insert_batch(conn, articles_batch, metadata_batch)
            dir_processed += len(articles_batch)
            
            # Verify a sample if needed
            if verify_sample > 0:
                verified = verify_batch(conn, articles_batch, verify_sample, log_file)
                dir_verified += verified
            
            # Delete JSON files if requested
            if delete_jsons:
                for i, (doc_id, _) in enumerate(articles_batch):
                    file_to_delete = os.path.join(directory, f"{doc_id}.json")
                    if os.path.exists(file_to_delete):
                        os.remove(file_to_delete)
        
        # Close connection
        conn.close()
        
        return dir_processed, dir_verified
    
    except Exception as e:
        log_message(log_file, f"Error processing directory {directory}: {str(e)}")
        return 0, 0

def insert_batch(conn, articles_batch, metadata_batch):
    """Insert batches of articles and metadata into the database"""
    with conn.cursor() as cur:
        # Insert articles
        execute_batch(cur, 
            "INSERT INTO articles (id, data) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
            articles_batch
        )
        
        # Insert metadata
        execute_batch(cur, 
            """
            INSERT INTO metadata (
                id, title, authors, date_download, date_modify, date_publish, 
                description, filename, image_url, language, localpath, 
                maintext, source_domain, title_page, title_rss, url
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                date_download = EXCLUDED.date_download,
                date_modify = EXCLUDED.date_modify,
                date_publish = EXCLUDED.date_publish,
                description = EXCLUDED.description,
                filename = EXCLUDED.filename,
                image_url = EXCLUDED.image_url,
                language = EXCLUDED.language,
                localpath = EXCLUDED.localpath,
                maintext = EXCLUDED.maintext,
                source_domain = EXCLUDED.source_domain,
                title_page = EXCLUDED.title_page,
                title_rss = EXCLUDED.title_rss,
                url = EXCLUDED.url
            """,
            metadata_batch
        )
        
        conn.commit()

def verify_batch(conn, articles_batch, verify_sample, log_file):
    """Verify a sample of the inserted articles"""
    verified_count = 0
    
    # Select a sample of articles to verify
    sample_size = max(1, int(len(articles_batch) * verify_sample))
    sample_indices = random.sample(range(len(articles_batch)), sample_size)
    
    with conn.cursor() as cur:
        for idx in sample_indices:
            doc_id, article_json = articles_batch[idx]
            
            # Query the database for this article
            cur.execute("SELECT data FROM articles WHERE id = %s", (doc_id,))
            result = cur.fetchone()
            
            if result:
                stored_json = result[0]
                original_dict = json.loads(article_json)
                
                # Compare the stored JSON with the original
                if stored_json == original_dict:
                    verified_count += 1
                else:
                    log_message(log_file, f"Verification failed for {doc_id}: content mismatch")
            else:
                log_message(log_file, f"Verification failed for {doc_id}: not found in database")
    
    return verified_count

def parallel_convert_directories(json_dir, db_params, max_workers=48, 
                               verify_sample=0.01, delete_jsons=False, batch_size=1000):
    """Convert JSON files to PostgreSQL in parallel"""
    start_time = time.time()
    
    # Create log file
    log_file = os.path.join(os.path.dirname(json_dir), "conversion_log.txt")
    log_message(log_file, f"Starting PostgreSQL conversion from {json_dir}")
    log_message(log_file, f"Database: {db_params['db_name']} on {db_params['host']}")
    log_message(log_file, f"Workers: {max_workers}, Verify: {verify_sample*100}%, Delete JSONs: {delete_jsons}")
    
    # Set up the database schema
    try:
        conn = get_db_connection(**db_params)
        setup_database(conn)
        conn.close()
        log_message(log_file, "Database schema setup complete")
    except Exception as e:
        log_message(log_file, f"Error setting up database schema: {str(e)}")
        return 0
    
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
    processed_dirs_path = os.path.join(os.path.dirname(json_dir), "processed_dirs.txt")
    
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
                db_params, 
                verify_sample, 
                delete_jsons,
                log_file,
                batch_size
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
    parser = argparse.ArgumentParser(description="Convert JSON files to PostgreSQL database")
    
    parser.add_argument('json_dir', type=str, 
                       help="Directory containing the JSON files to convert")
    
    parser.add_argument('--db_name', type=str, default="news_articles",
                       help="PostgreSQL database name (default: news_articles)")
    
    parser.add_argument('--db_user', type=str, default="postgres",
                       help="PostgreSQL user name (default: postgres)")
    
    parser.add_argument('--db_password', type=str, required=True,
                       help="PostgreSQL password")
    
    parser.add_argument('--db_host', type=str, default="localhost",
                       help="PostgreSQL host (default: localhost)")
    
    parser.add_argument('--db_port', type=int, default=5432,
                       help="PostgreSQL port (default: 5432)")
    
    parser.add_argument('--workers', type=int, default=48,
                       help="Number of parallel workers to use (default: 48)")
    
    parser.add_argument('--verify', type=float, default=0.01,
                       help="Fraction of documents to verify (default: 0.01 = 1%%)")
    
    parser.add_argument('--delete', action='store_true',
                       help="Delete JSON files after successful conversion")
    
    parser.add_argument('--batch_size', type=int, default=1000,
                       help="Batch size for database inserts (default: 1000)")
    
    args = parser.parse_args()
    
    # Database connection parameters
    db_params = {
        'db_name': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'host': args.db_host,
        'port': args.db_port
    }
    
    # Convert JSON files to PostgreSQL
    parallel_convert_directories(
        args.json_dir,
        db_params,
        max_workers=args.workers,
        verify_sample=args.verify,
        delete_jsons=args.delete,
        batch_size=args.batch_size
    )