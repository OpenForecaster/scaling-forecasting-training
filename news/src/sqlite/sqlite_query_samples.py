import sqlite3
import os
import json
import lmdb

def filter_by_date_download(output_dir, date_download):
    """Find all articles with a specific date_download"""
    sqlite_path = os.path.join(output_dir, "metadata.db")
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM article_metadata WHERE date_download = ?", 
        (date_download,)
    )
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results

def filter_by_date_range(output_dir, start_date, end_date):
    """Find all articles within a date range"""
    sqlite_path = os.path.join(output_dir, "metadata.db")
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM article_metadata WHERE date_publish BETWEEN ? AND ?", 
        (start_date, end_date)
    )
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return results

def get_article_content(output_dir, doc_id, shard=None):
    """Get full article content by ID"""
    # If shard is not provided, look it up in SQLite
    if shard is None:
        sqlite_path = os.path.join(output_dir, "metadata.db")
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT shard FROM article_metadata WHERE id = ?", (doc_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        shard = result[0]
    
    # Open the appropriate shard
    shard_path = os.path.join(output_dir, f'shard_{shard}')
    env = lmdb.open(shard_path, readonly=True, max_dbs=3)
    articles_db = env.open_db(b'articles')
    
    # Get the article
    with env.begin(db=articles_db) as txn:
        value = txn.get(doc_id.encode('utf-8'))
    
    env.close()
    
    if value:
        return json.loads(value.decode('utf-8'))
    return None