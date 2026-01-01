import os
import argparse
import logging
import json
from typing import List, Dict
import sys
import copy
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from qgen.article_processor import ArticleProcessor

def main():
    parser = argparse.ArgumentParser(description="Select and filter news articles for forecasting question generation")
    
    # Article input
    parser.add_argument(
        "--article_path", 
        type=str, 
        default="/fast/sgoel/forecasting/news/tokenized_data/news/www.apnews.com_tokenized.jsonl",
        help="Path to news article file or directory"
    )
    
    # Selection parameters
    parser.add_argument(
        "--filter_years",
        type=int,
        nargs="+",
        default=None,
        help="Filter articles by year(s) (e.g., --filter_years 2021 2022)"
    )
    parser.add_argument(
        "--random_sample",
        type=int,
        nargs="+",
        default=None,
        help="Number(s) of articles to randomly sample from the filtered list (e.g., --random_sample 100 500 1000)"
    )

    parser.add_argument(
        "--balance_yearly",
        action="store_true",
        help="Balance article selection across years when filtering or sampling"
    )
    
    parser.add_argument(
        "--hard_cutoff",
        type=str,
        default="2025-11",
        help="Exclude articles published on or after this date (format: YYYY-MM)"
    )
    
    parser.add_argument(
        "--pre_cutoff",
        type=str,
        default="2020-01",
        help="Exclude articles published before this date (format: YYYY-MM)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit the number of articles to process"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save selected articles (defaults to same directory as input)"
    )
    
    parser.add_argument(
        "--filter_language",
        type=str,
        default="en", # Default to English
        help="Filter articles by language code during loading (e.g., 'en'). Set to 'None' or empty string to disable."
    )
    
    parser.add_argument(
        "--min_word_count",
        type=int,
        default=100,
        help="Minimum word count required for articles (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Handle 'None' string from command line for language filter
    filter_language = None if args.filter_language and args.filter_language.lower() == 'none' else args.filter_language

    # Load articles, applying language filter during load
    processor = ArticleProcessor(args.article_path)
    processor.load_articles(limit=args.limit, language_code=filter_language) # Pass language filter here
    if args.filter_years:
        processor.filter_by_years(args.filter_years)
    
    # Apply hard cutoff if specified
    if args.hard_cutoff:
        processor.filter_by_date_cutoff(args.hard_cutoff)
        
    # Apply pre-cutoff if specified
    if args.pre_cutoff:
        processor.filter_by_date_cutoff(args.pre_cutoff, reverse=True)
    
    # Apply minimum word count filter
    if args.min_word_count > 0:
        processor.filter_by_min_word_count(args.min_word_count, log_stats=True)
    
    # Determine input filename and directory for output path construction
    input_filename = os.path.basename(args.article_path)
    input_dir = os.path.dirname(args.article_path)
    
    # Extract base name without extension
    base_name = os.path.splitext(input_filename)[0]
    if base_name.endswith("_tokenized"):
        base_name = base_name[:-10]  # Remove "_tokenized" suffix
    
    # Use specified output directory or default to input directory
    output_dir = args.output_dir if args.output_dir else input_dir
    
    # Create qgen_selected subdirectory
    qgen_selected_dir = os.path.join(output_dir, "qgen_selected")
    os.makedirs(qgen_selected_dir, exist_ok=True)
    
    # If no random sampling is specified, just save the filtered articles
    if not args.random_sample:
        # Create output filename
        output_filename = f"{base_name}_{args.hard_cutoff}_selected{len(processor.articles)}.jsonl"
        output_path = os.path.join(qgen_selected_dir, output_filename)
        
        # Save selected articles
        with open(output_path, 'w') as f:
            for article in processor.articles:
                f.write(json.dumps(article) + '\n')
        
        logger.info(f"Selected {len(processor.articles)} articles and saved to {output_path}")
    else:
        # For each sample size, create a separate file
        for sample_size in tqdm(args.random_sample, desc="Sampling articles"):
            # Apply random sampling for this specific size
            sampled_articles = processor.random_sample(sample_size, balance_yearly=args.balance_yearly)
            
            # Create output filename with _selected<count> suffix
            output_filename = f"{base_name}_{args.hard_cutoff}_selected{sample_size}.jsonl"
            output_path = os.path.join(qgen_selected_dir, output_filename)
            
            # Save selected articles
            with open(output_path, 'w') as f:
                for article in sampled_articles:
                    f.write(json.dumps(article) + '\n')
            
            logger.info(f"Selected {len(sampled_articles)} articles (sample size: {sample_size}) and saved to {output_path}")
    
    # Print summary of selection
    logger.info("Selection summary:")
    if args.filter_years:
        logger.info(f"  - Filtered by years: {args.filter_years}")
    if args.hard_cutoff:
        logger.info(f"  - Hard cutoff date: {args.hard_cutoff}")
    if args.min_word_count > 0:
        logger.info(f"  - Minimum word count: {args.min_word_count}")
    if args.random_sample:
        logger.info(f"  - Random sample sizes: {args.random_sample}")
    if args.limit:
        logger.info(f"  - Article limit: {args.limit}")
    if args.balance_yearly:
        logger.info(f"  - Balanced across years: Yes")
    if filter_language:
        logger.info(f"  - Filtered by language during load: {filter_language}")
    else:
        logger.info(f"  - Language filtering disabled.")

if __name__ == "__main__":
    main()