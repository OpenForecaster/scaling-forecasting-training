# retrieval_system.py
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
from pathlib import Path
import dateutil.parser
from datasets import load_dataset, Dataset, load_from_disk
import re
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# ---------------------- Date Processing Module ----------------------

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string into datetime object."""
    if not date_str or date_str == "None" or date_str is None:
        return None
    try:
        return dateutil.parser.parse(date_str)
    except (ValueError, TypeError):
        return None

def is_before_cutoff(article_date: Optional[datetime], cutoff_date: Optional[datetime]) -> bool:
    """Check if article date is before cutoff date."""
    if not article_date or not cutoff_date:
        return False
    return article_date < cutoff_date

# ---------------------- Date Strategy Pattern ----------------------

class DateCutoffStrategy:
    """Base class for date cutoff strategies."""
    def get_cutoff_date(self, article: Dict[str, Any]) -> Optional[datetime]:
        """Return the cutoff date for an article."""
        raise NotImplementedError

class DaysBeforeDateCutoffStrategy(DateCutoffStrategy):
    """Strategy for getting a date that is X days before the latest date."""
    def __init__(self, days_before: int = 30):
        self.days_before = days_before
    
    def get_cutoff_date(self, article: Dict[str, Any]) -> Optional[datetime]:
        dates = []
        for date_field in ['date_download_parsed', 'date_modify_parsed', 'date_publish_parsed']:
            if date_field in article and article[date_field]:
                dates.append(article[date_field])
        
        if not dates:
            return None
        
        latest_date = max(dates)
        return latest_date - timedelta(days=self.days_before)

# ---------------------- Data Loading Module ----------------------

def load_jsonl_articles(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load articles from a JSONL file."""
    articles = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line)
                # Parse dates in advance
                for date_field in ['date_download', 'date_modify', 'date_publish']:
                    if date_field in article and article[date_field]:
                        article[f"{date_field}_parsed"] = parse_date(article[date_field])
                articles.append(article)
            except json.JSONDecodeError:
                continue
    return articles

def load_metaculus_dataset(path: str = "nikhilchandak/metaculus-binary") -> Dataset:
    """Load the Metaculus dataset from Hugging Face or local disk."""
    # Check if path is a local directory (saved with save_to_disk)
    if os.path.isdir(path):
        return load_from_disk(path)
    else:
        # Load from Hugging Face
        return load_dataset(path)["train"]

# ---------------------- BM25 Indexing Module ----------------------

class BM25Index:
    def __init__(self):
        self.corpus = []  # List of document texts
        self.articles = []  # Original article data
        self.tokenized_corpus = []  # Tokenized documents
        self.index = None  # BM25 index
        
    def prepare_document(self, article: Dict[str, Any]) -> str:
        """Prepare document text for indexing from article."""
        # Use pre-tokenized news if available
        if 'tokenized_news' in article:
            return ' '.join(article['tokenized_news'])
        
        else:
            raise ValueError("No tokenized news found in article")
    
    def add_articles(self, articles: List[Dict[str, Any]]) -> None:
        """Add articles to the index."""
        for article in articles:
            doc_text = self.prepare_document(article)
            self.corpus.append(doc_text)
            self.articles.append(article)
            self.tokenized_corpus.append(article['tokenized_news'])
        
        # Rebuild the index
        self.build_index()
    
    def build_index(self) -> None:
        """Build BM25 index from tokenized corpus."""
        print(f"Building index from {len(self.tokenized_corpus)} documents")
        if self.tokenized_corpus:
            self.index = BM25Okapi(self.tokenized_corpus)
    
    def save_index(self, filepath: str) -> None:
        """Save the index and related data to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'corpus': self.corpus,
                'articles': self.articles,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
    
    def load_index(self, filepath: str) -> None:
        """Load index and related data from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.corpus = data['corpus']
            self.articles = data['articles']
            self.tokenized_corpus = data['tokenized_corpus']
        self.build_index()

# ---------------------- Retrieval Module ----------------------

class RetrievalSystem:
    def __init__(self, index_provider: Any):
        """
        Initialize retrieval system with an index provider.
        The index_provider should have methods for searching and accessing articles.
        """
        self.index_provider = index_provider
    
    def prepare_query(self, background: str, resolution_criteria: str) -> str:
        """Prepare query from background and resolution criteria."""
        return f"{background} {resolution_criteria}"
    
    def tokenize_query(self, query: str) -> List[str]:
        """Tokenize the query for retrieval."""
        if isinstance(query, str):
            # Return query as is, assuming it's already tokenized elsewhere
            return query.split()
        return query
    
    def retrieve(self, 
                query: str, 
                top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve articles based on query.
        
        Args:
            query: The query text
            top_k: Number of articles to retrieve
            
        Returns:
            List of retrieved articles with scores
        """
        if not self.index_provider.index:
            return []
        
        tokenized_query = self.tokenize_query(query)
        scores = self.index_provider.index.get_scores(tokenized_query)
        
        # Use faster NumPy approach to get top k articles
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build the results
        results = []
        for idx in top_indices:
            results.append({
                'article': self.index_provider.articles[idx],
                'score': scores[idx],
                'index': idx
            })
        
        return results

# ---------------------- Date-Aware Index Manager ----------------------

class DateIndexManager:
    """Manages a BM25 index that efficiently handles date filtering by maintaining a moving index."""
    
    def __init__(self, articles=None, d_freshness=1):
        """
        Initialize the date-aware index manager.
        
        Args:
            articles: List of article dictionaries (optional)
            d_freshness: Number of days difference before refreshing the index
        """
        self.all_articles = pd.DataFrame()
        self.current_index = BM25Index()
        self.d_freshness = d_freshness
        self.last_cutoff_date = None
        
        if articles:
            self.add_articles(articles)
    
    def add_articles(self, articles):
        """Add articles to the index and convert them to a DataFrame with date_final column."""
        # Convert articles to DataFrame
        articles_df = pd.DataFrame(articles)
        
        # Create date_final column as the max of all date fields
        date_columns = ['date_publish_parsed', 'date_download_parsed', 'date_modify_parsed']
        
        # Initialize date_final as None
        articles_df['date_final'] = None
        
        # For each article, find the latest date among all date fields
        for i, row in articles_df.iterrows():
            dates = []
            for date_field in date_columns:
                if date_field in row and row[date_field] is not None:
                    # Convert to timezone-naive datetime if it has a timezone
                    dt = row[date_field]
                    if dt is not None:
                        # Skip if not a valid datetime type
                        if not isinstance(dt, (datetime, pd.Timestamp)):
                            continue
                        # Ensure dt is a datetime object
                        if isinstance(dt, pd.Timestamp):
                            dt = dt.to_pydatetime()
                        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                        dates.append(dt)
            
            if dates:
                articles_df.at[i, 'date_final'] = max(dates)
        
        # Append to existing DataFrame
        self.all_articles = pd.concat([self.all_articles, articles_df], ignore_index=True)
    
    def refresh_index_for_date(self, cutoff_date, lookback_days=365):
        """
        Refresh the index to include only articles before the cutoff date.
        
        Args:
            cutoff_date: Only include articles from before this date
            lookback_days: How many days before cutoff_date to include
        
        Returns:
            True if index was refreshed, False if using cached index
        """
        # Check if we can reuse the existing index (within freshness period)
        if (self.last_cutoff_date and cutoff_date and 
            abs((cutoff_date - self.last_cutoff_date).days) <= self.d_freshness):
            return False
        
        # Calculate the earliest date to include
        if cutoff_date and lookback_days:
            earliest_date = cutoff_date - timedelta(days=lookback_days)
        else:
            earliest_date = None
        
        # Filter articles by date using DataFrame query
        filtered_df = self.all_articles
        
        if cutoff_date:
            filtered_df = filtered_df[filtered_df['date_final'] <= cutoff_date]
        
        if earliest_date:
            filtered_df = filtered_df[filtered_df['date_final'] >= earliest_date]
        
        # Rebuild the index with filtered articles
        self.current_index = BM25Index()
        self.current_index.add_articles(filtered_df.to_dict('records'))
        self.last_cutoff_date = cutoff_date
        
        return True

# ---------------------- Main Workflow ----------------------

class NewsRetrievalWorkflow:
    def __init__(self, 
                date_strategy: DateCutoffStrategy = None,
                d_freshness: int = 1,
                lookback_days: int = 365):
        """
        Initialize news retrieval workflow.
        
        Args:
            date_strategy: Strategy for determining cutoff dates
            d_freshness: Number of days difference before refreshing the index
            lookback_days: How many days before cutoff_date to include in index
        """
        self.date_strategy = date_strategy or DaysBeforeDateCutoffStrategy()
        self.lookback_days = lookback_days
        # Create a DateIndexManager to handle date-based filtering
        self.index_manager = DateIndexManager(d_freshness=d_freshness)
        
    def load_articles_from_paths(self, paths: List[str]) -> None:
        """Load articles from multiple paths (files or directories) and add them to the index manager."""
        articles_list = []
        for path in paths:
            articles = load_jsonl_articles(path)
            articles_list.extend(articles)
        
        # Add articles to the index manager if we have any
        if articles_list:
            self.index_manager.add_articles(articles_list)
        
        print(f"Loaded {len(self.index_manager.all_articles)} articles in total")
    
    def retrieve_for_metaculus_question(self, 
                                      question: Dict[str, Any], 
                                      top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve articles for a single Metaculus question.
        
        Args:
            question: A Metaculus question dictionary
            top_k: Number of articles to retrieve
            
        Returns:
            List of retrieved articles with scores or empty list if no valid date
        """
        # Use just the question text as the query
        query = question.get('joined_query', '')
        
        # Check for valid resolution date - return empty if none
        if 'date_resolve_at' not in question or not question['date_resolve_at']:
            return []
            
        date_resolve_at = parse_date(question['date_resolve_at'])
        if not date_resolve_at:
            return []
            
        # Apply the days_before strategy to get the actual cutoff date
        cutoff_date = date_resolve_at - timedelta(days=self.date_strategy.days_before)
        
        # Use the DateIndexManager to refresh the index for this cutoff date
        self.index_manager.refresh_index_for_date(cutoff_date, self.lookback_days)
        
        # Create temporary retrieval system with the date-filtered index
        temp_retrieval = RetrievalSystem(self.index_manager.current_index)
        
        # Get results using the date-filtered index
        retrieved = temp_retrieval.retrieve(query, top_k=top_k)
        return retrieved
    
    def retrieve_for_metaculus(self, 
                               dataset_path: str = "nikhilchandak/metaculus-binary", 
                               top_k: int = 10,
                               output_file: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve articles for Metaculus questions.
        
        Args:
            dataset_path: Path to the Metaculus dataset
            top_k: Number of articles to retrieve per question
            output_file: Path to save trend statistics
            
        Returns:
            Dictionary mapping question IDs to retrieved articles
        """
        ds = load_metaculus_dataset(dataset_path)
        results = {}
        
        # Track questions with insufficient articles
        insufficient_articles_count = 0
        skipped_questions_count = 0
        
        # For trend analysis
        trend_data = {}
        
        # Create a list of just question IDs and resolution dates for efficient sorting
        question_dates = []
        for i in range(len(ds[list(ds.features.keys())[0]])):
            question_id = str(i)
            date_resolve_at = ds['date_resolve_at'][i] if 'date_resolve_at' in ds.features else None
            
            # Skip questions without dates
            if not date_resolve_at:
                skipped_questions_count += 1
                continue
                
            parsed_date = parse_date(date_resolve_at)
            if not parsed_date:
                skipped_questions_count += 1
                continue
                
            question_dates.append((question_id, parsed_date))
        
        print(f"Skipping {skipped_questions_count} questions without valid resolution dates")
        
        # Sort questions by resolution date
        question_dates.sort(key=lambda x: x[1])
        
        # Process questions in sorted order
        for q_idx, (question_id, question_date) in tqdm(enumerate(question_dates), total=len(question_dates), desc="Processing questions"):
            # Get the full question data
            i = int(question_id)
            question = {key: ds[key][i] for key in ds.features.keys()}
            
            # Load tokenized query if available
            if 'tokenized_query' in question:
                query_tokens = question['tokenized_query']
                question['joined_query'] = ' '.join(query_tokens)
            
            filtered_results = self.retrieve_for_metaculus_question(question, top_k)
            
            # Get index size for this question
            index_size = len(self.index_manager.current_index.articles) if self.index_manager.current_index else 0
            # print(f"Question {question_id}: Date = {question_date}, Index size = {index_size}, Retrieved articles = {len(filtered_results)}")
            
            # Collect trend data
            highest_score = max([r['score'] for r in filtered_results]) if filtered_results else 0
            mean_score = np.mean([r['score'] for r in filtered_results]) if filtered_results else 0
            trend_data[question_id] = {
                'date': question_date,
                'index_size': index_size,
                'highest_score': highest_score,
                'mean_score': mean_score,
                'num_retrieved': len(filtered_results)
            }
            
            # Random sampling with 1% probability 
            if np.random.random() < 0.01:
                print(f"\nRandom sample - Question {question_id}:")
                print(f"  Question text: {question.get('question', 'No question text')}")
                print(f"  Resolution date: {question.get('date_resolve_at', 'Unknown')}")
                
                cutoff_date = None
                if 'date_resolve_at' in question and question['date_resolve_at']:
                    date_resolve_at = parse_date(question['date_resolve_at'])
                    if date_resolve_at:
                        cutoff_date = date_resolve_at - timedelta(days=self.date_strategy.days_before)
                print(f"  Cutoff date (date_resolve_at - {self.date_strategy.days_before} days): {cutoff_date}")
                
                print(f"  Retrieved {len(filtered_results)} out of {top_k} requested articles")
                for j, result in enumerate(filtered_results):
                    article = result['article']
                    print(f"  {j+1}. {article.get('title', 'No title')} (Score: {result['score']:.3f})")
                    print(f"     Date: {article.get('date_publish', 'Unknown')}")
                    print(f"     URL: {article.get('url', 'Unknown')}")
            
            # Check if we have fewer than top_k articles
            if len(filtered_results) < top_k:
                insufficient_articles_count += 1
            
            results[question_id] = filtered_results
        
        # Save trend data if output_file is provided
        if output_file:
            trend_path = os.path.join(output_file, "trends.pkl")
            os.makedirs(os.path.dirname(trend_path), exist_ok=True)
            with open(trend_path, 'wb') as f:
                pickle.dump(trend_data, f)
            print(f"Trend data saved to {trend_path}")
        
        print(f"Number of questions with fewer than {top_k} articles retrieved: {insufficient_articles_count}")
        return results, insufficient_articles_count

# ---------------------- Example Usage ----------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="News retrieval system for forecasting questions")
    # Data loading arguments
    parser.add_argument("--jsonl_paths", type=str, default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/apnews.com_tokenized.jsonl", 
                        help="Path to JSONL file or directory containing news articles")
    parser.add_argument("--metaculus_dataset", type=str, default="/fast/sgoel/forecasting/news/tokenized_data/questions/metaculus-binary_tokenized",
                        help="Path or name of the Metaculus dataset")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save retrieval results (Hugging Face dataset format)")
    
    # Retrieval arguments
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of articles to retrieve per question")
    parser.add_argument("--days_before", type=int, default=30,
                        help="Number of days before question date to use as cutoff")
    parser.add_argument("--d_freshness", type=int, default=1,
                        help="Number of days difference before refreshing the index")
    parser.add_argument("--lookback_days", type=int, default=365,
                        help="Number of days to look back from cutoff date for articles")
    
    # Output arguments
    parser.add_argument("--sample_size", type=int, default=3,
                        help="Number of sample results to print")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the workflow with date strategy
    date_strategy = DaysBeforeDateCutoffStrategy(days_before=args.days_before)
    workflow = NewsRetrievalWorkflow(
        date_strategy=date_strategy,
        d_freshness=args.d_freshness,
        lookback_days=args.lookback_days
    )
    
    # Process jsonl_paths - convert to list of paths
    jsonl_path = args.jsonl_paths
    jsonl_paths_list = []
    
    path_obj = Path(jsonl_path)
    if path_obj.is_file():
        # Single file
        jsonl_paths_list = [jsonl_path]
    elif path_obj.is_dir():
        # Directory - get all JSONL files
        jsonl_paths_list = [str(p) for p in path_obj.glob('**/*.jsonl')]
        print(f"Found {len(jsonl_paths_list)} JSONL files in directory {jsonl_path}")
    else:
        print(f"Warning: Path {jsonl_path} is neither a file nor a directory")
    
    # Load articles from paths (files or directories)
    print(f"Loading articles from {jsonl_path}")
    workflow.load_articles_from_paths(jsonl_paths_list)
    
    # Load the Metaculus dataset
    print(f"Loading Metaculus questions from {args.metaculus_dataset}")
    original_dataset = load_metaculus_dataset(args.metaculus_dataset)
    
    # Retrieve for Metaculus questions
    print(f"Retrieving articles for Metaculus questions from {jsonl_path}")
    results, insufficient_count = workflow.retrieve_for_metaculus(
        dataset_path=args.metaculus_dataset,
        top_k=args.top_k,
        output_file=args.output_file  # Pass output_file for saving trend data
    )
    
    # Save results in Hugging Face dataset format if output file is specified
    if args.output_file:
        print(f"Creating augmented dataset and saving to {args.output_file}")
        
        # Prepare the retrieved articles data
        retrieved_articles_data = []
        for i in range(len(original_dataset)):
            question_id = str(i)
            retrieved = results.get(question_id, [])
            
            # Format articles for storage in the dataset
            formatted_articles = []
            for result in retrieved:
                article_data = {
                    'score': float(result['score']),
                    'index': int(result['index']),
                    'title': result['article'].get('title', ''),
                    'maintext': result['article'].get('maintext', ''),
                    'source_domain': result['article'].get('source_domain', ''),
                    'url': result['article'].get('url', ''),
                    'language': result['article'].get('language', ''),
                    'author': result['article'].get('author', ''),
                    'date_download': result['article'].get('date_download', ''),
                    'date_modify': result['article'].get('date_modify', ''),
                    'date_publish': result['article'].get('date_publish', ''),
                }
                formatted_articles.append(article_data)
            
            retrieved_articles_data.append(formatted_articles)
        
        # Create a new dataset with the additional column
        new_dataset = original_dataset.add_column("retrieved_articles", retrieved_articles_data)
        
        # Save the augmented dataset
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        new_dataset.save_to_disk(args.output_file)
        print(f"Dataset saved to {args.output_file}")
        print(f"Number of questions with fewer than {args.top_k} articles retrieved: {insufficient_count}")
    


if __name__ == "__main__":
    main()