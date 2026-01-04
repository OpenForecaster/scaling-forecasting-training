#!/usr/bin/env python3
"""
Filter Articles CLI - Command-line interface for article filtering and selection.

This script provides a unified interface for filtering articles by relevance
and selecting subsets based on various criteria.

Usage:
    # Filter for relevance
    python scripts/filter_articles_cli.py \\
        --articles_path articles.jsonl \\
        --model_path /path/to/model \\
        --filter_articles
    
    # Select and sample articles
    python scripts/filter_articles_cli.py \\
        --article_path articles.jsonl \\
        --filter_years 2021 2022 \\
        --random_sample 1000 \\
        --min_word_count 100 \\
        --output_dir output/
"""

import os
import argparse
import logging
import sys

# Add parent of qgen directory to path so we can import qgen package
qgen_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(qgen_dir)
sys.path.insert(0, parent_dir)

from qgen.filters.article_filter import ArticleFilter, ArticleSelector

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Filter and select news articles")
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["filter", "select"],
        required=True,
        help="Mode: 'filter' for relevance filtering, 'select' for article selection"
    )
    
    # Common arguments
    parser.add_argument(
        "--article_path",
        type=str,
        required=True,
        help="Path to article file or directory"
    )
    
    # Filtering arguments
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model for relevance filtering"
    )
    parser.add_argument(
        "--filter_articles",
        action="store_true",
        help="Filter articles for forecasting relevance"
    )
    parser.add_argument(
        "--filter_questions",
        action="store_true",
        help="Filter questions for forecasting relevance"
    )
    
    # Selection arguments
    parser.add_argument(
        "--filter_years",
        type=int,
        nargs="+",
        help="Filter articles by year(s)"
    )
    parser.add_argument(
        "--hard_cutoff",
        type=str,
        help="Exclude articles on or after this date (YYYY-MM)"
    )
    parser.add_argument(
        "--pre_cutoff",
        type=str,
        help="Exclude articles before this date (YYYY-MM)"
    )
    parser.add_argument(
        "--min_word_count",
        type=int,
        default=100,
        help="Minimum word count for articles"
    )
    parser.add_argument(
        "--random_sample",
        type=int,
        nargs="+",
        help="Sample size(s) for random sampling"
    )
    parser.add_argument(
        "--balance_yearly",
        action="store_true",
        help="Balance sampling across years"
    )
    parser.add_argument(
        "--filter_language",
        type=str,
        default="en",
        help="Filter by language code"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory for output files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Specific output file path (for filter mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "filter":
        # Relevance filtering mode
        if not args.model_path:
            logger.error("--model_path required for filter mode")
            return
        
        filter = ArticleFilter(model_path=args.model_path)
        articles = filter.load_articles(args.article_path)
        
        if args.filter_articles:
            articles = filter.evaluate_articles(articles)
        
        if args.filter_questions:
            articles = filter.evaluate_questions(articles)
        
        output_path = args.output_path or args.article_path.replace('.jsonl', '_filtered.jsonl')
        filter.save_articles(articles, output_path, args.filter_articles, args.filter_questions)
        
    elif args.mode == "select":
        # Selection mode
        selector = ArticleSelector(args.article_path)
        
        # Apply filters
        filter_language = None if args.filter_language and args.filter_language.lower() == 'none' else args.filter_language
        selector.load_articles(language_code=filter_language)
        
        if args.filter_years:
            selector.filter_by_years(args.filter_years)
        
        if args.hard_cutoff:
            selector.filter_by_date_cutoff(args.hard_cutoff)
        
        if args.pre_cutoff:
            selector.filter_by_date_cutoff(args.pre_cutoff, reverse=True)
        
        if args.min_word_count > 0:
            selector.filter_by_min_word_count(args.min_word_count, log_stats=True)
        
        # Sample and save
        if args.random_sample:
            for sample_size in args.random_sample:
                sampled = selector.random_sample(sample_size, balance_yearly=args.balance_yearly)
                output_path = os.path.join(args.output_dir or ".", f"selected_{sample_size}.jsonl")
                selector.save_selected(sampled, output_path)
        else:
            # Save all filtered articles
            articles = selector.get_articles()
            output_path = os.path.join(args.output_dir or ".", "selected.jsonl")
            selector.save_selected(articles, output_path)


if __name__ == "__main__":
    main()


