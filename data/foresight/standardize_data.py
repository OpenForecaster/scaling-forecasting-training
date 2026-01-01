#!/usr/bin/env python3
"""
OpenForesight Dataset Standardization Script

Purpose:
    Standardizes forecasting question datasets for OpenForesight benchmark.
    Ensures consistent date formats, removes data leakage, and generates prompts.

Main Features:
    - Standardize all date fields to YYYY-MM-DD format
    - Ensure question_start_date <= resolution_date - 1 day
    - Filter by cutoff_date for temporal evaluation
    - Generate prompts with and without retrieval
    - Remove questions where answer appears in question/criteria
    - Build retrieved articles summaries for RAG

Output Format:
    JSONL with standardized fields including:
    - question_title, background, resolution_criteria
    - answer, answer_type
    - date fields (standardized)
    - prompt (with retrieval), prompt_without_retrieval

Usage:
    python standardize_data.py --input_file questions.jsonl --output_file standardized.jsonl --cutoff_date 2025-01-01
"""

import json
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """
    Load all data from JSONL or JSON file.
    
    Args:
        file_path: Path to the JSONL or JSON file
        
    Returns:
        List of dictionaries with all data from the file
    """
    questions_data = []

    # Determine file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    try:
                        article = json.loads(line.strip())
                        # article['idx'] = line_idx
                        questions_data.append(article)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_idx}: {e}")
                        continue

    elif ext == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, dict):
                    # Check if it's a numbered key structure (like {"1": {...}, "2": {...}})
                    if all(key.isdigit() for key in data.keys()):
                        # Convert to list format
                        articles = []
                        for key in sorted(data.keys(), key=int):
                            article = data[key]
                            # article['idx'] = len(articles)
                            articles.append(article)
                        data = articles
                    else:
                        # Single object, wrap in list
                        data = [data]
                
                for line_idx, article in enumerate(data):
                    # article['idx'] = line_idx
                    questions_data.append(article)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON file {file_path}: {e}")
    else:
        logger.error(f"Unsupported file extension for {file_path}. Only .jsonl and .json are supported.")
        return []

    logger.info(f"Loaded {len(questions_data)} items from {file_path}")
    return questions_data


def format_forecasting_prompt_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
    extra_info2 = ""
    extra_info1 = ""
    if len(retrieved_news_articles_summaries) > 10:
        extra_info1 = " You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer."
        extra_info2 = f"\nRelevant passages from retrieved news articles:\n{retrieved_news_articles_summaries}\n"
    
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer.{extra_info1} Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}
{extra_info2}
Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt


def standardize_date(date_value) -> Optional[str]:
    """
    Standardize date format to YYYY-MM-DD.
    Handles multiple date formats including timestamps, ISO formats, and various string formats.
    
    Args:
        date_value: Can be int (timestamp), str (various formats), or None
        
    Returns:
        Date string in YYYY-MM-DD format, or None if conversion fails
    """
    if date_value is None or date_value == "":
        return None
    
    # If it's a numeric timestamp, convert it first
    if isinstance(date_value, (int, float)):
        try:
            dt = datetime.fromtimestamp(date_value)
            return dt.strftime('%Y-%m-%d')
        except (ValueError, OSError, OverflowError):
            pass
    
    # Try to convert string to numeric timestamp first
    if isinstance(date_value, str):
        try:
            timestamp = int(float(date_value))
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%Y-%m-%d')
        except (ValueError, TypeError, OSError, OverflowError):
            pass
    
    # Handle different date string formats
    if isinstance(date_value, str):
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z'
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_value, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try to handle Z suffix with fromisoformat
        if date_value.endswith('Z'):
            try:
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass
        
        # Try fromisoformat as a fallback
        try:
            dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    # If all else fails, log warning and return None
    logger.warning(f"Could not convert date value: {date_value} (type: {type(date_value)})")
    return None


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string in YYYY-MM-DD format to datetime object.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.warning(f"Could not parse date: {date_str}")
        return None


def build_retrieved_articles_summaries(relevant_docs: List, num_articles: int = 10) -> str:
    """
    Build retrieved news articles summaries string from relevant_docs.
    Similar to how it's done in eval_retrieval.py
    
    Args:
        relevant_docs: List of relevant documents/articles
        num_articles: Number of articles to include
        
    Returns:
        Formatted string with article summaries
    """
    retrieved_news_articles_summaries = ""
    
    j = 1
    for doc in relevant_docs[:num_articles]:
        article_title = None
        article_summary = None
        article_passage = None
        article_date = None
        article_source = None
        source_text = ""
        date_text = ""
        
        for item in doc:
            if isinstance(item, dict):
                if "title" in item:
                    article_title = item["title"]
                
                if "relevant_passage" in item:
                    article_passage = item["relevant_passage"]
                
                elif "summary" in item and item.get("prompt_name") == "create_forecast_summarization_prompt":
                    article_summary = item["summary"]
                
                if "max_date" in item:
                    article_date = item["max_date"]
                    # Convert timestamp to human readable format
                    if isinstance(article_date, (int, float)):
                        article_date = datetime.fromtimestamp(article_date).strftime("%B %d, %Y")
                        date_text = f"Article Publish Date: {article_date}\n"
                
                if "source_domain" in item:
                    article_source = item["source_domain"]
                    source_text = f"Source: {article_source}\n"
        
        if article_title is not None:
            if article_passage is not None:
                retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\n{source_text}{date_text}Relevant Passage: {article_passage}\n\n"
            elif article_summary is not None:
                retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\n{source_text}{date_text}Summary: {article_summary}\n\n"
        
        j += 1
    
    return retrieved_news_articles_summaries


def standardize_data(
    input_file: str,
    output_file: str,
    cutoff_date: Optional[str] = None,
    num_articles: int = 10
) -> None:
    """
    Standardize questions data file by:
    1. Removing 'relevant_articles_sorted_by_docs' attribute
    2. Converting dates to YYYY-MM-DD format
    3. Ensuring question_start_date <= resolution_date - 1 day
    4. Filtering by cutoff_date
    5. Adding 'prompt' and 'prompt_without_retrieval' columns
    
    Args:
        input_file: Path to input JSONL/JSON file
        output_file: Path to output JSONL file
        cutoff_date: Date string in YYYY-MM-DD format to filter by (resolution_date < cutoff_date)
        num_articles: Number of articles to use for retrieval in prompt
    """
    logger.info(f"Loading questions from: {input_file}")
    questions_data = load_questions_from_jsonl(input_file)
    
    if not questions_data:
        logger.error("No valid questions found in the input file")
        return
    
    logger.info(f"Loaded {len(questions_data)} questions")
    
    # Parse cutoff_date if provided
    cutoff_datetime = None
    if cutoff_date:
        cutoff_datetime = parse_date(cutoff_date)
        if cutoff_datetime is None:
            logger.error(f"Invalid cutoff_date format: {cutoff_date}. Expected YYYY-MM-DD")
            return
        logger.info(f"Filtering out questions with resolution_date < {cutoff_date} (keeping questions with resolution_date >= {cutoff_date})")
    
    standardized_data = []
    filtered_count = 0
    filtered_answer_in_text_count = 0
    date_conversion_errors = 0
    max_resolution_date = "2025-01-01"
    
    for row in questions_data:
        # Filter out samples where answer appears in question_title or resolution_criteria
        answer = row.get('answer', '')
        question_title = row.get('question_title', '')
        background = row.get('background', '')
        resolution_criteria = row.get('resolution_criteria', '')
        
        if answer:
            # Case-insensitive check if answer appears in question_title or resolution_criteria
            answer_lower = str(answer).lower().strip()
            question_title_lower = str(question_title).lower()
            background_lower = str(background).lower()
            resolution_criteria_lower = str(resolution_criteria).lower()
            
            if answer_lower in question_title_lower or answer_lower in resolution_criteria_lower or answer_lower in background_lower:
                filtered_answer_in_text_count += 1
                continue
        # STEP 1: Generate prompts first using original data (before any modifications)
        # Get relevant_docs from original row for prompt generation
        relevant_docs = row.get('relevant_docs', row.get('relevant_articles_sorted_by_docs', []))
        
        # Handle both list and JSON string formats for relevant_docs
        if isinstance(relevant_docs, str):
            try:
                relevant_docs = json.loads(relevant_docs)
            except json.JSONDecodeError:
                relevant_docs = []
        
        # Build retrieved articles summaries for prompt generation
        retrieved_news_articles_summaries = build_retrieved_articles_summaries(
            relevant_docs if isinstance(relevant_docs, list) else [],
            num_articles=num_articles
        )
        
        # Generate prompt with retrieval (using original row data)
        prompt_with_retrieval = format_forecasting_prompt_with_retrieval(
            question_title=row.get('question_title', ''),
            background=row.get('background', ''),
            resolution_criteria=row.get('resolution_criteria', ''),
            answer_type=row.get('answer_type', ''),
            retrieved_news_articles_summaries=retrieved_news_articles_summaries,
        )
        
        # Generate prompt without retrieval (empty retrieved_news_articles_summaries)
        prompt_without_retrieval = format_forecasting_prompt_with_retrieval(
            question_title=row.get('question_title', ''),
            background=row.get('background', ''),
            resolution_criteria=row.get('resolution_criteria', ''),
            answer_type=row.get('answer_type', ''),
            retrieved_news_articles_summaries="",  # Empty string for no retrieval
        )
        
        # STEP 2: Now create standardized row and make modifications
        # Create a copy of the row
        standardized_row = row.copy()
        
        # Remove 'relevant_articles_sorted_by_docs' attribute
        if 'relevant_articles_sorted_by_docs' in standardized_row:
            del standardized_row['relevant_articles_sorted_by_docs']
        
        # Remove 'relevant_docs' attribute
        if 'relevant_docs' in standardized_row:
            del standardized_row['relevant_docs']
        
        # Remove 'original_file' attribute
        if 'original_file' in standardized_row:
            del standardized_row['original_file']
        
        # Standardize all date columns to YYYY-MM-DD format
        date_columns = [
            'resolution_date',
            'question_start_date',
            'article_publish_date',
            'article_modify_date',
            'article_download_date',
            'date_publish',
            'article_date_publish'
        ]
        
        resolution_date_str = None
        question_start_date_str = None
        
        for col in date_columns:
            if col in standardized_row:
                date_value = standardized_row[col]
                standardized_date = standardize_date(date_value)
                if standardized_date:
                    standardized_row[col] = standardized_date
                    # Track these for later use
                    if col == 'resolution_date':
                        resolution_date_str = standardized_date
                    elif col == 'question_start_date':
                        question_start_date_str = standardized_date
                else:
                    if date_value:
                        date_conversion_errors += 1
                        logger.warning(f"Could not convert {col}: {date_value}")
                    standardized_row[col] = ''
        
        # Ensure question_start_date <= resolution_date - 1 day
        if resolution_date_str and question_start_date_str:
            # assert resolution date is before 2025-05-01
            if resolution_date_str >= "2025-05-01":
                logger.error(f"Resolution date {resolution_date_str} is before 2025-05-01 for row: {standardized_row}")
                raise ValueError(f"Resolution date {resolution_date_str} is before 2025-05-01 for row: {standardized_row}")
            
            max_resolution_date = max(max_resolution_date, resolution_date_str)
            resolution_dt = parse_date(resolution_date_str)
            question_start_dt = parse_date(question_start_date_str)
            
            if resolution_dt and question_start_dt:
                # Calculate resolution_date - 1 day
                max_question_start = resolution_dt - timedelta(days=1)
                
                # If question_start_date is after resolution_date - 1 day, adjust it
                if question_start_dt > max_question_start:
                    standardized_row['question_start_date'] = max_question_start.strftime("%Y-%m-%d")
                    logger.debug(f"Adjusted question_start_date to {standardized_row['question_start_date']} "
                               f"(was {question_start_date_str}, resolution_date: {resolution_date_str})")
        
        # Filter by cutoff_date if provided
        # Filter out rows where resolution_date < cutoff_date (keep rows where resolution_date >= cutoff_date)
        if cutoff_datetime and resolution_date_str:
            resolution_dt = parse_date(resolution_date_str)
            if resolution_dt and resolution_dt < cutoff_datetime:
                filtered_count += 1
                continue
        
        # Add the prompts to standardized row
        standardized_row['prompt'] = prompt_with_retrieval
        standardized_row['prompt_without_retrieval'] = prompt_without_retrieval
        
        standardized_data.append(standardized_row)
    
    logger.info(f"Max resolution date: {max_resolution_date}")
    logger.info(f"Processed {len(standardized_data)} questions (filtered {filtered_count} by cutoff_date)")
    logger.info(f"Filtered {filtered_answer_in_text_count} samples where answer appears in question_title or resolution_criteria")
    if date_conversion_errors > 0:
        logger.warning(f"Encountered {date_conversion_errors} date conversion errors")
    
    # Print column names in the final file
    if standardized_data:
        # Get all unique column names from all rows
        all_columns = set()
        for row in standardized_data:
            all_columns.update(row.keys())
        column_names = sorted(list(all_columns))
        logger.info(f"Column names in final file ({len(column_names)} columns): {', '.join(column_names)}")
    else:
        logger.warning("No data to save, cannot determine column names")
    
    
    # Assert question_start_date < resolution_date for all rows
    for row in standardized_data:
        question_start_date = row.get('question_start_date', '')
        resolution_date = row.get('resolution_date', '')
        if question_start_date and resolution_date:
            # print(f"Question start date: {question_start_date}, Resolution date: {resolution_date}")
            question_start_date_dt = parse_date(question_start_date)
            resolution_date_dt = parse_date(resolution_date)
            if question_start_date_dt and resolution_date_dt and question_start_date_dt > resolution_date_dt:
                logger.error(f"Question start date {question_start_date} is after resolution date {resolution_date} for row: {row}")
                raise ValueError(f"Question start date {question_start_date} is after resolution date {resolution_date} for row: {row}")
            
    # Save to output file
    logger.info(f"Saving standardized data to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in standardized_data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(standardized_data)} standardized questions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize questions data file")
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to input JSONL/JSON file containing questions"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help="Path to output JSONL file. If not provided, defaults to /fast/nchandak/forecasting/openforesight/<input_filename>"
    )
    parser.add_argument(
        '--cutoff_date',
        type=str,
        default=None,
        help="Cutoff date in YYYY-MM-DD format. Questions with resolution_date >= cutoff_date will be filtered out"
    )
    parser.add_argument(
        '--num_articles',
        type=int,
        default=5,
        help="Number of articles to use for retrieval in prompt generation"
    )
    
    args = parser.parse_args()
    
    # Set default output_file if not provided
    if args.output_file is None:
        input_basename = os.path.basename(args.input_file)
        args.output_file = os.path.join("/fast/nchandak/forecasting/openforesight", input_basename)
        logger.info(f"Output file not specified, using default: {args.output_file}")
    
    standardize_data(
        input_file=args.input_file,
        output_file=args.output_file,
        cutoff_date=args.cutoff_date,
        num_articles=args.num_articles
    )

