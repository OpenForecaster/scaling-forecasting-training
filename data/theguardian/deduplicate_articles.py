import json
import os
from datetime import datetime

def deduplicate_jsonl(input_file_path):
    """
    Remove duplicate articles from JSONL file based on unique article IDs and URLs
    """
    print(f"Processing file: {input_file_path}")
    
    if not os.path.exists(input_file_path):
        print(f"Error: File {input_file_path} does not exist!")
        return
    
    seen_ids = set()
    seen_urls = set()
    unique_articles = []
    total_articles = 0
    duplicates_by_id = 0
    duplicates_by_url = 0
    
    # Read and process the JSONL file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line.strip())
                total_articles += 1
                
                article_id = article.get('id', '')
                article_url = article.get('url', '')
                
                # Check for duplicates by ID
                if article_id in seen_ids:
                    duplicates_by_id += 1
                    print(f"Duplicate ID found (line {line_num}): {article_id}")
                    continue
                
                # Check for duplicates by URL  
                if article_url in seen_urls:
                    duplicates_by_url += 1
                    print(f"Duplicate URL found (line {line_num}): {article_url}")
                    continue
                
                # Article is unique, add to our collections
                seen_ids.add(article_id)
                seen_urls.add(article_url)
                unique_articles.append(article)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_path = input_file_path.replace('.jsonl', f'_deduplicated_{timestamp}.jsonl')
    
    # Save deduplicated articles
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for article in unique_articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DEDUPLICATION SUMMARY")
    print("="*60)
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Total articles processed: {total_articles:,}")
    print(f"Duplicates by ID: {duplicates_by_id}")
    print(f"Duplicates by URL: {duplicates_by_url}")
    print(f"Total duplicates removed: {duplicates_by_id + duplicates_by_url}")
    print(f"Unique articles saved: {len(unique_articles):,}")
    print(f"Duplicate rate: {((duplicates_by_id + duplicates_by_url) / total_articles * 100):.2f}%")
    
    # File size comparison
    input_size = os.path.getsize(input_file_path)
    output_size = os.path.getsize(output_file_path)
    print(f"Input file size: {input_size / (1024*1024):.1f} MB")
    print(f"Output file size: {output_size / (1024*1024):.1f} MB")
    print(f"Size reduction: {((input_size - output_size) / input_size * 100):.1f}%")
    
    return output_file_path, len(unique_articles), duplicates_by_id + duplicates_by_url

def validate_uniqueness(file_path):
    """
    Validate that the deduplicated file actually contains no duplicates
    """
    print(f"\nValidating uniqueness in: {file_path}")
    
    seen_ids = set()
    seen_urls = set()
    article_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line.strip())
                article_count += 1
                
                article_id = article.get('id', '')
                article_url = article.get('url', '')
                
                if article_id in seen_ids:
                    print(f"ERROR: Duplicate ID still found: {article_id}")
                    return False
                
                if article_url in seen_urls:
                    print(f"ERROR: Duplicate URL still found: {article_url}")
                    return False
                
                seen_ids.add(article_id)
                seen_urls.add(article_url)
                
            except Exception as e:
                print(f"Error validating line {line_num}: {e}")
                return False
    
    print(f"✅ Validation successful: {article_count:,} unique articles confirmed")
    return True

def main():
    """
    Main function to deduplicate Guardian articles
    """
    # Path to the original file
    input_file = "/fast/nchandak/forecasting/newsdata/theguardian/www.theguardian.com_july2025_20250731_181157.jsonl"
    
    print("Guardian Articles Deduplication Tool")
    print("=" * 50)
    
    try:
        # Perform deduplication
        output_file, unique_count, duplicates_removed = deduplicate_jsonl(input_file)
        
        # Validate the result
        if validate_uniqueness(output_file):
            print(f"\n✅ Deduplication completed successfully!")
            print(f"📁 Deduplicated file: {output_file}")
            
            # Optionally replace original file with deduplicated version
            replace_original = input("Replace original file with deduplicated version? (y/n): ").lower().strip()
            if replace_original == 'y':
                import shutil
                # Backup original
                backup_file = input_file.replace('.jsonl', '_backup.jsonl')
                shutil.copy2(input_file, backup_file)
                print(f"📋 Original file backed up to: {backup_file}")
                
                # Replace with deduplicated version
                shutil.move(output_file, input_file)
                print(f"✅ Original file replaced with deduplicated version")
            
        else:
            print("❌ Validation failed!")
            
    except Exception as e:
        print(f"❌ Error during deduplication: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 