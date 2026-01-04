import json
import random
import argparse
import os
from datetime import datetime

def load_articles(file_path):
    """
    Load all articles from JSONL file
    """
    articles = []
    print(f"Loading articles from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                article = json.loads(line.strip())
                articles.append(article)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(articles):,} articles")
    return articles

def sample_articles(articles, num_articles, seed=None):
    """
    Randomly sample specified number of articles
    """
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    if num_articles > len(articles):
        print(f"Warning: Requested {num_articles:,} articles but only {len(articles):,} available")
        print(f"Returning all {len(articles):,} articles")
        return articles.copy()
    
    if num_articles <= 0:
        print("Error: Number of articles must be positive")
        return []
    
    print(f"Randomly sampling {num_articles:,} articles from {len(articles):,} total articles")
    sampled = random.sample(articles, num_articles)
    
    return sampled

def save_sampled_articles(articles, output_path):
    """
    Save sampled articles to JSONL file
    """
    print(f"Saving {len(articles):,} articles to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')
    
    file_size = os.path.getsize(output_path)
    print(f"File saved successfully: {file_size / (1024*1024):.1f} MB")

def print_sample_statistics(original_articles, sampled_articles):
    """
    Print statistics about the sampling
    """
    print("\n" + "="*60)
    print("SAMPLING STATISTICS")
    print("="*60)
    print(f"Original articles: {len(original_articles):,}")
    print(f"Sampled articles: {len(sampled_articles):,}")
    print(f"Sampling rate: {(len(sampled_articles) / len(original_articles) * 100):.2f}%")
    
    # Count by section for original
    original_sections = {}
    for article in original_articles:
        # Extract section from the article data
        section = "Unknown"
        if 'url' in article:
            url_parts = article['url'].split('/')
            if len(url_parts) > 3:
                section = url_parts[3]  # Extract section from URL
        original_sections[section] = original_sections.get(section, 0) + 1
    
    # Count by section for sampled
    sampled_sections = {}
    for article in sampled_articles:
        section = "Unknown"
        if 'url' in article:
            url_parts = article['url'].split('/')
            if len(url_parts) > 3:
                section = url_parts[3]
        sampled_sections[section] = sampled_sections.get(section, 0) + 1
    
    print(f"\nTop 10 sections in sample:")
    sorted_sections = sorted(sampled_sections.items(), key=lambda x: x[1], reverse=True)
    for section, count in sorted_sections[:10]:
        original_count = original_sections.get(section, 0)
        percentage = (count / original_count * 100) if original_count > 0 else 0
        print(f"  {section}: {count} articles ({percentage:.1f}% of original {original_count})")

def main():
    """
    Main function to sample Guardian articles
    """
    parser = argparse.ArgumentParser(description='Randomly sample Guardian articles from JSONL file')
    parser.add_argument('--num_articles', '-n', type=int, required=True,
                      help='Number of articles to randomly sample')
    parser.add_argument('--input_file', '-i', type=str,
                      default='/fast/nchandak/forecasting/newsdata/theguardian/www.theguardian.com_20250731.jsonl',
                      help='Path to input JSONL file (default: deduplicated July 2025 file)')
    parser.add_argument('--output_dir', '-o', type=str,
                      default='/fast/nchandak/forecasting/newsdata/theguardian/',
                      help='Output directory for sampled file')
    parser.add_argument('--seed', '-s', type=int, default=None,
                      help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    print("Guardian Articles Sampling Tool")
    print("=" * 50)
    print(f"Input file: {args.input_file}")
    print(f"Number of articles to sample: {args.num_articles:,}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    
    try:
        # Load all articles
        articles = load_articles(args.input_file)
        
        if not articles:
            print("No articles loaded. Exiting.")
            return
        
        # Sample articles
        sampled_articles = sample_articles(articles, args.num_articles, args.seed)
        
        if not sampled_articles:
            print("No articles sampled. Exiting.")
            return
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_basename = os.path.basename(args.input_file).replace('.jsonl', '')
        output_filename = f"{input_basename}_sample_{args.num_articles}.jsonl"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save sampled articles
        save_sampled_articles(sampled_articles, output_path)
        
        # Print statistics
        print_sample_statistics(articles, sampled_articles)
        
        print(f"\n✅ Sampling completed successfully!")
        print(f"📁 Sampled file: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 