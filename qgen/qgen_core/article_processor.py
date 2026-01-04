"""
Article Processor - Load, filter, and sample news articles for question generation.

This module provides the ArticleProcessor class for managing large collections of
news articles. It handles:

1. **Article Loading**: Load from single files or directories of JSONL files
2. **Date-based Filtering**: Filter articles by publication year and date ranges
3. **Quality Filtering**: Filter by word count and other quality metrics
4. **Language Filtering**: Select articles in specific languages
5. **Random Sampling**: Sample articles with optional yearly balancing
6. **Statistics Tracking**: Track article counts by year and other metadata

The processor is designed to handle large datasets efficiently with progress bars
and incremental processing capabilities.

Key Features:
- Efficient loading from single files or entire directories
- Multiple filtering criteria (dates, word count, language)
- Balanced sampling across years
- Comprehensive statistics and logging
- Memory-efficient processing for large datasets

Example Usage:
    ```python
    from qgen.qgen_core.article_processor import ArticleProcessor
    
    # Initialize processor with article path
    processor = ArticleProcessor("/path/to/articles/")
    
    # Load articles with language filter
    processor.load_articles(language_code="en")
    
    # Apply filters
    processor.filter_by_years([2021, 2022, 2023])
    processor.filter_by_date_cutoff("2024-01")
    processor.filter_by_min_word_count(100)
    
    # Sample articles with yearly balancing
    sampled = processor.random_sample(1000, balance_yearly=True)
    ```

Author: Forecasting Team
"""

import os
import json
import glob
import logging
import random
from typing import List, Dict, Counter
from tqdm import tqdm  # Add tqdm import

logger = logging.getLogger(__name__)

class ArticleProcessor:
    def __init__(self, article_path: str):
        """
        Initialize the article processor.
        
        Args:
            article_path: Path to article file or directory containing article files
        """
        self.article_path = article_path
        self.articles = []
        self.year_stats = Counter()  # Track count of articles by year
        self.articles_by_year = {}   # Store articles grouped by year
        
    def _extract_date_components(self, date_str: str):
        """
        Extract year and month from a date string.
        
        Args:
            date_str: Date string in format 'YYYY-MM-DD' or similar
            
        Returns:
            Tuple of (year, month) or None if date is invalid
        """
        if not date_str or not isinstance(date_str, str) or len(date_str) < 7:
            return None
        
        try:
            year = int(date_str[:4])
            month = int(date_str[5:7])
            return (year, month)
        except (ValueError, IndexError):
            return None

    def _compute_max_date(self, article: Dict) -> Dict:
        """
        Compute the maximum date from date_publish, date_modify, and date_download.
        Adds max_date field to the article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Updated article dictionary with max_date field
        """
        # Get all date fields
        date_publish = article.get('date_publish', '')
        date_modify = article.get('date_modify', '')
        date_download = article.get('date_download', '')
        
        # Date publish seems to be faulty in many cases, so we'll use date_download instead (since date_modify is often NULL)
        date_publish = date_download
        # date_download = max([date_publish, date_modify])
        
        # Filter valid dates (ensure they're proper strings with at least YYYY-MM format)
        valid_dates = []
        for date_str in [date_publish, date_modify, date_download]:
            if isinstance(date_str, str) and len(date_str) >= 7:
                try:
                    # Basic validation - check if first parts can be parsed as numbers
                    int(date_str[:4])  # Year
                    int(date_str[5:7])  # Month
                    valid_dates.append(date_str)
                except (ValueError, IndexError):
                    pass
        
        # Set max date if we have valid dates
        if valid_dates:
            # String comparison works because dates are in YYYY-MM-DD format
            article['max_date'] = max(valid_dates)
            # Extract year for year-based filtering and statistics
            article['max_date_year'] = int(article['max_date'][:4])
        else:
            article['max_date'] = None
            article['max_date_year'] = None
        
        return article

    def load_articles(self, limit: int = None, language_code: str = None) -> List[Dict]:
        """
        Load articles from the specified path, optionally filtering by language.
        If path is a directory, load all JSONL files in the directory.
        If path is a file, load the single file.
        
        Args:
            limit: Maximum number of articles to load (for debugging)
            language_code: Language code to keep (e.g., 'en'). If None, no language filtering.
        
        Returns:
            List of article dictionaries
        """
        # Reset year stats and articles by year when loading new articles
        self.year_stats = Counter()
        self.articles_by_year = {}
        self.articles = [] # Reset articles list too
        
        if os.path.isdir(self.article_path):
            # Get all JSONL files in the directory
            jsonl_files = glob.glob(os.path.join(self.article_path, "*.jsonl"))
            logger.info(f"Found {len(jsonl_files)} JSONL files in directory {self.article_path}")
            
            # Use tqdm for progress bar over files
            for file_path in tqdm(jsonl_files, desc="Loading files"):
                self._load_articles_from_file(file_path, limit, language_code) # Pass language_code
                if limit and len(self.articles) >= limit:
                    break  # Stop processing files once we hit the limit
        else:
            # Load a single file
            self._load_articles_from_file(self.article_path, limit, language_code) # Pass language_code
            
        logger.info(f"Total number of articles fetched: {len(self.articles)}")
        if language_code:
             logger.info(f"Filtered for language: '{language_code}'")
        logger.info(f"Articles per year: {dict(self.year_stats)}")
        return self.articles
    
    def filter_by_years(self, years: List[int]) -> List[Dict]:
        """
        Filter articles by max date year.
        
        Args:
            years: List of years to include
            
        Returns:
            Filtered list of article dictionaries
        """
        if not years:
            return self.articles
            
        filtered_articles = []
        
        # Just combine all articles from the filtered years
        for year in years:
            if year in self.articles_by_year:
                filtered_articles.extend(self.articles_by_year[year])
        
        # Remove years not in years from self.articles_by_year
        self.articles_by_year = {year: self.articles_by_year[year] for year in years if year in self.articles_by_year}
        
        # Update year stats
        self.year_stats = Counter()
        for year, articles in self.articles_by_year.items():
            self.year_stats[year] = len(articles)
        
        logger.info(f"Filtered to {len(filtered_articles)} articles from years {years}")
        self.articles = filtered_articles
        return self.articles

    def filter_by_date_cutoff(self, cutoff_date: str, reverse: bool = False) -> List[Dict]:
        """
        Filter out articles with max date on or after the cutoff date.
        
        Args:
            cutoff_date: Cutoff date in format 'YYYY-MM'
            
        Returns:
            Filtered list of article dictionaries
        """
        if not cutoff_date:
            return self.articles
        
        # Ensure cutoff_date is in a format that can be compared (YYYY-MM becomes YYYY-MM-01)
        comparison_cutoff = cutoff_date
        if len(cutoff_date) == 7:  # If only YYYY-MM is provided
            comparison_cutoff = f"{cutoff_date}-01"  # Add day for proper comparison
            
        # Get years affected by the cutoff
        try:
            cutoff_year = int(cutoff_date[:4])
            affected_years = [year for year in self.articles_by_year if year >= cutoff_year]
            
            # Filter articles within each affected year
            for year in affected_years:
                filtered_year_articles = [
                    article for article in self.articles_by_year[year]
                    if article['max_date'] is None or (article['max_date'] < comparison_cutoff if not reverse else article['max_date'] > comparison_cutoff)
                ]
                self.articles_by_year[year] = filtered_year_articles
                self.year_stats[year] = len(filtered_year_articles)
            
            # Rebuild self.articles from articles_by_year
            self.articles = []
            for year_articles in self.articles_by_year.values():
                self.articles.extend(year_articles)
            
            logger.info(f"Filtered to {len(self.articles)} articles with max date before {cutoff_date} with reverse={reverse}")
            return self.articles
            
        except ValueError:
            logger.error(f"Invalid cutoff date format: {cutoff_date}. Expected format: YYYY-MM")
            return self.articles

    def filter_by_min_word_count(self, min_word_count: int = 20, log_stats: bool = False) -> List[Dict]:
        """
        Filter out articles with word count below the specified minimum.
        
        Args:
            min_word_count: Minimum number of words required in the article
            log_stats: If True, log word count statistics before and after filtering
            
        Returns:
            Filtered list of article dictionaries
        """
        if not min_word_count or min_word_count <= 0:
            return self.articles
        
        original_count = len(self.articles)
        
        # Log statistics before filtering if requested
        if log_stats:
            before_stats = self.get_word_count_stats()
            logger.info(f"Word count statistics before filtering: {before_stats}")
        
        filtered_articles = []
        filtered_count = 0
        
        # Filter articles by word count
        for article in self.articles:
            # Calculate word count from maintext and description
            word_count = 0
            maintext = article.get('maintext', '')
            description = article.get('description', '')
            
            if maintext:
                word_count += len(maintext.split())
            if description:
                word_count += len(description.split())
            
            if word_count >= min_word_count:
                filtered_articles.append(article)
            else:
                filtered_count += 1
        
        # Update articles list
        self.articles = filtered_articles
        
        # Rebuild articles_by_year and year_stats
        self.articles_by_year = {}
        self.year_stats = Counter()
        
        for article in self.articles:
            year = article.get('max_date_year')
            if year is not None:
                if year not in self.articles_by_year:
                    self.articles_by_year[year] = []
                self.articles_by_year[year].append(article)
                self.year_stats[year] += 1
        
        # Log statistics after filtering if requested
        if log_stats:
            after_stats = self.get_word_count_stats()
            logger.info(f"Word count statistics after filtering: {after_stats}")
        
        logger.info(f"Filtered to {len(self.articles)} articles with at least {min_word_count} words (removed {filtered_count} articles)")
        return self.articles

    def get_word_count_stats(self) -> Dict:
        """
        Calculate word count statistics for all articles.
        
        Returns:
            Dictionary containing word count statistics
        """
        if not self.articles:
            return {
                'total_articles': 0,
                'min_words': 0,
                'max_words': 0,
                'avg_words': 0,
                'median_words': 0
            }
        
        word_counts = []
        for article in self.articles:
            word_count = 0
            maintext = article.get('maintext', '')
            description = article.get('description', '')
            
            if maintext:
                word_count += len(maintext.split())
            if description:
                word_count += len(description.split())
            
            word_counts.append(word_count)
        
        word_counts.sort()
        
        stats = {
            'total_articles': len(self.articles),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'avg_words': sum(word_counts) / len(word_counts),
            'median_words': word_counts[len(word_counts) // 2]
        }
        
        return stats

    def random_sample(self, sample_size: int, balance_yearly: bool = False) -> List[Dict]:
        """
        Randomly sample articles based on a target sample size.
        
        Args:
            sample_size: Target number of articles to sample
            balance_yearly: If True, balance samples across years
            
        Returns:
            Sampled list of article dictionaries
        """
        if not sample_size or sample_size >= len(self.articles):
            return self.articles
        
        if not balance_yearly:
            # Use random.sample for efficient random sampling without replacement
            sampled_articles = random.sample(self.articles, min(sample_size, len(self.articles)))
        else:
            # Get years that have articles in the current filtered set without iterating through all articles
            years_with_articles = [year for year in self.articles_by_year.keys() 
                                if year in self.year_stats and self.year_stats[year] > 0]
            
            if not years_with_articles:
                logger.warning("No articles with valid years found, using standard random sampling")
                return random.sample(self.articles, min(sample_size, len(self.articles)))
            
            # Initialize available articles for each year
            available_articles = {}
            for year in years_with_articles:
                available_articles[year] = list(self.articles_by_year[year])
                random.shuffle(available_articles[year])  # Shuffle for randomness
            
            # Active years for round-robin sampling
            active_years = list(years_with_articles)
            
            # Sample articles round-robin from each year with progress bar
            sampled_articles = []
            pbar = tqdm(total=sample_size, desc="Sampling articles")
            while len(sampled_articles) < sample_size and active_years:
                # Loop through all active years
                years_to_remove = []
                for year in list(active_years):
                    # Check if we've reached the sample size
                    if len(sampled_articles) >= sample_size:
                        break
                    
                    # If articles are available for this year, take one
                    if available_articles[year]:
                        article = available_articles[year].pop()
                        sampled_articles.append(article)
                        pbar.update(1)  # Update progress bar
                        
                        # If no more articles for this year, mark it for removal
                        if not available_articles[year]:
                            years_to_remove.append(year)
                    else:
                        years_to_remove.append(year)
                
                # Remove years with no more articles
                for year in years_to_remove:
                    if year in active_years:
                        active_years.remove(year)
            
            pbar.close()  # Close the progress bar
            
            # Log the sampling results
            year_counts = {}
            for article in sampled_articles:
                year = article['max_date_year']
                year_counts[year] = year_counts.get(year, 0) + 1
            
            for year, count in sorted(year_counts.items()):
                logger.info(f"Sampled {count} articles from year {year}")
        
        logger.info(f"Randomly sampled {len(sampled_articles)} articles (target: {sample_size})")
        return sampled_articles
    
    def _load_articles_from_file(self, file_path: str, limit: int = None, language_code: str = None) -> None:
        """
        Load articles from a JSONL file, optionally filtering by language.

        Args:
            file_path: Path to the JSONL file
            limit: Maximum number of articles to load
            language_code: Language code to keep (e.g., 'en'). If None, no language filtering.
        """
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist")
            return

        with open(file_path, 'r') as f:
            count = 0
            count_no_max_date = 0
            count_lang_filtered = 0 # Track articles filtered by language
            for line in f:
                try:
                    article = json.loads(line.strip())

                    # --- Language Filter ---
                    # Apply language filter *before* processing dates or adding to lists
                    if language_code and article.get('language') != language_code:
                        count_lang_filtered += 1
                        continue # Skip this article

                    # Compute and add max date fields
                    article = self._compute_max_date(article)

                    # Only add articles with valid max dates
                    if article['max_date'] is not None:
                        self.articles.append(article)

                        # Track year statistics and group articles by year using max date
                        year = article['max_date_year']
                        self.year_stats[year] += 1

                        # Add to articles_by_year dictionary
                        if year not in self.articles_by_year:
                            self.articles_by_year[year] = []
                        self.articles_by_year[year].append(article)

                        count += 1

                        # Check if we've reached the limit (based on successfully loaded articles)
                        if limit and len(self.articles) >= limit:
                            logger.info(f"Reached article limit ({limit}), stopping")
                            break
                    else:
                        count_no_max_date += 1

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {file_path}")

            logger.info(f"Loaded {count} articles from {file_path}")
            if language_code:
                logger.info(f"Filtered out {count_lang_filtered} articles based on language ('{language_code}')")
            # logger.info(f"Articles per year: {dict(self.year_stats)}") # This might be too verbose here, logged in load_articles
            logger.info(f"Skipped {count_no_max_date} articles with no max date")