# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess forecasting dataset and save to JSONL format.

This script loads forecasting questions from JSONL files, formats them with
retrieved news articles, and saves the processed dataset in a standard JSONL
format that can be easily loaded and viewed.
"""

import argparse
import logging
import os
import sys
import json
from typing import List, Dict, Any
from datetime import datetime
from verl.utils.hdfs_io import copy, makedirs
import random

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    force=True,  # Force reconfiguration if already configured
)
logger = logging.getLogger(__name__)
# Also configure root logger to ensure output
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Ensure output is not buffered
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass  # Ignore if reconfigure fails


def format_forecasting_prompt_binary(
    question_title: str,
    background: str,
    resolution_criteria: str,
) -> str:
    """
    Format a binary forecasting prompt without article context.
    
    Binary questions ask whether an event will happen (YES/NO) and require
    a probability estimate between 0 and 1.
    
    Args:
        question_title: The title of the forecasting question
        background: Background information about the question
        resolution_criteria: Criteria for how the question will be resolved
        
    Returns:
        Formatted prompt string for binary forecasting
    """
    prompt = f"""You will be asked a binary forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt


def is_date_after_cutoff(date_str: str, cutoff_date: str = "2025-05-01") -> bool:
    """
    Check if a date is after the cutoff date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        cutoff_date: Cutoff date in YYYY-MM-DD format
        
    Returns:
        True if date is after cutoff, False otherwise
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        cutoff_obj = datetime.strptime(cutoff_date, '%Y-%m-%d')
        return date_obj > cutoff_obj
    except ValueError:
        return False


def format_forecasting_prompt(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
) -> str:
    """
    Format a prompt for single outcome forecasting (non-binary questions).
    
    These questions require a specific answer (not just YES/NO) along with
    a confidence probability.
    
    Args:
        question_title: The title of the forecasting question
        background: Background information about the question
        resolution_criteria: Criteria for how the question will be resolved
        answer_type: Expected type of answer (e.g., "number", "date", "text")
        
    Returns:
        Formatted prompt string for single outcome forecasting
    """
    prompt = f"""You will be asked a forecasting question (which might be from the past). You have to come up with the best guess for the final answer. Please provide your reasoning before stating your final answer and also express how likely you think your answer is to be correct (your confidence in your answer).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Expected Answer Type: {answer_type}

Think step by step about the information provided, reason about uncertainty and put your final answer (in the format asked) in <answer> </answer> tags. You should also specify your confidence in your answer in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt


def format_forecasting_prompt_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    answer_type: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """
    Format a prompt for single outcome forecasting with retrieved news articles.
    
    Similar to format_forecasting_prompt but includes relevant news articles
    that may help answer the question.
    
    Args:
        question_title: The title of the forecasting question
        background: Background information about the question
        resolution_criteria: Criteria for how the question will be resolved
        answer_type: Expected type of answer
        retrieved_news_articles_summaries: Formatted summaries of retrieved articles
        
    Returns:
        Formatted prompt string with article context
    """
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


def format_forecasting_prompt_binary_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """
    Format a binary forecasting prompt with retrieved news articles.
    
    Similar to format_forecasting_prompt_binary but includes relevant news articles.
    
    Args:
        question_title: The title of the forecasting question
        background: Background information about the question
        resolution_criteria: Criteria for how the question will be resolved
        retrieved_news_articles_summaries: Formatted summaries of retrieved articles
        
    Returns:
        Formatted prompt string for binary forecasting with article context
    """
    extra_info2 = ""
    extra_info1 = ""
    if len(retrieved_news_articles_summaries) > 10:
        extra_info1 = " You will also be provided with a list of retrieved news articles summaries which you may refer to when coming up with your answer."
        extra_info2 = f"\nRelevant passages from retrieved news articles:\n{retrieved_news_articles_summaries}\n"
    
    prompt = f"""You will be asked a binary forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened.{extra_info1} Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {question_title}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
{extra_info2}
Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags."""

    return prompt


def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """
    Load forecasting questions from a JSONL file.
    
    Filters out irrelevant questions and articles, extracts question components,
    and normalizes date formats.
    
    Args:
        file_path: Path to the JSONL file containing question data
        
    Returns:
        List of dictionaries, each containing question information
    """
    questions_data = []
    logger.info(f"Loading questions from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    article = json.loads(line.strip())
                    
                    # Skip if question is not relevant or the article is not relevant
                    if 'question_relevant' in article and int(article['question_relevant']) == 0:
                        continue
                    
                    if 'article_relevant' in article and int(article['article_relevant']) == 0:
                        continue
                    
                    if 'no_good_question' in article and int(article['no_good_question']) == 1:
                        continue
                    
                    # Extract resolution from answer if not explicitly provided
                    resolution = article.get('resolution', -1)
                    qanswer = article.get('answer', '')
                    if "yes" in qanswer.lower():
                        resolution = 1
                    elif "no" in qanswer.lower():
                        resolution = 0
                    else:
                        resolution = -1
                        
                    # Create a question entry with all necessary fields
                    question_entry = {
                        'idx': line_idx,
                        'question_title': article.get('question_title', article.get('question', '')),
                        'background': article.get('background', ''),
                        'resolution_criteria': article.get('resolution_criteria', ''),
                        'answer': qanswer,
                        'answer_type': article.get('answer_type', ''),
                        'resolution_date': article.get('resolution_date', article.get('date_resolve_at', '')),
                        'question_start_date': article.get('question_start_date', article.get('date_begin', '')),
                        'question_idx': article.get('question_idx', ''),
                        'relevant_docs': article.get('relevant_articles_sorted_by_docs', article.get('relevant_docs', [])),
                        'url': article.get('url', ''),
                        'data_source': article.get('data_source', 'unknown'),
                        'news_source': article.get('news_source', 'unknown'),
                        'resolution': resolution,
                        'prompt': article.get('prompt', ''),
                    }
                    
                    # Convert timestamp dates to YYYY-MM-DD format
                    if 'resolution_date' in question_entry:
                        if isinstance(question_entry['resolution_date'], int):
                            question_entry["resolution_date"] = datetime.fromtimestamp(
                                question_entry['resolution_date']
                            ).strftime('%Y-%m-%d')
                            
                    if 'question_start_date' in question_entry:
                        if isinstance(question_entry['question_start_date'], int):
                            question_entry['question_start_date'] = datetime.fromtimestamp(
                                question_entry['question_start_date']
                            ).strftime('%Y-%m-%d')
                        
                    # Only add if we have a valid question title
                    if question_entry['question_title'].strip():
                        questions_data.append(question_entry)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data


def format_retrieved_articles(relevant_docs: List[Dict], num_articles: int) -> str:
    """
    Format retrieved news articles into a readable string format.
    
    Extracts title, source, date, authors, and relevant passages/summaries
    from each article and formats them for inclusion in prompts.
    
    Args:
        relevant_docs: List of article dictionaries containing article information
        num_articles: Maximum number of articles to include
        
    Returns:
        Formatted string containing article summaries
    """
    retrieved_news_articles_summaries = ""
    
    for j in range(1, min(len(relevant_docs) + 1, num_articles + 1)):
        doc = relevant_docs[j - 1]
        
        article_title = None
        article_summary = None
        article_passage = None
        source_text = ""
        date_text = ""
        author_text = ""
        
        # Extract information from document structure
        for item in doc:
            if isinstance(item, dict):
                if "title" in item:
                    article_title = item["title"]
                    
                if "relevant_passage" in item:
                    article_passage = item["relevant_passage"]
                elif "summary" in item and item.get("prompt_name") == "create_forecast_summarization_prompt":
                    article_summary = item["summary"]
                    
                if "source_domain" in item:
                    article_source = item["source_domain"]
                    source_text = f"Source: {article_source}\n"
                        
                if "max_date" in item:
                    article_date = item["max_date"]
                    # Convert timestamp to human-readable format
                    article_date = datetime.fromtimestamp(article_date).strftime("%B %d, %Y") 
                    date_text = f"Article Publish Date: {article_date}\n"
                    
                if "authors" in item and len(item["authors"]) > 0:
                    first5authors = item["authors"][:5]
                    first5 = ",".join(first5authors)
                    author_text = f"Article Author(s): {first5}\n"
                    
        # Format the article entry
        if article_title is not None:
            if article_passage is not None:
                retrieved_news_articles_summaries += (
                    f"Article {j}:\nTitle: {article_title}\n{source_text}{author_text}"
                    f"{date_text}Relevant Passage: {article_passage}\n\n"
                )
            elif article_summary is not None:
                retrieved_news_articles_summaries += (
                    f"Article {j}:\nTitle: {article_title}\n{source_text}{author_text}"
                    f"{date_text}Summary: {article_summary}\n\n"
                )
    
    return retrieved_news_articles_summaries


def process_question(
    example: Dict[str, Any],
    split: str,
    idx: int,
    cutoff_date: str = "2024-05-01"
) -> Dict[str, Any]:
    """
    Process a single question into the final dataset format.
    
    This function:
    1. Determines how many articles to retrieve (0-5, randomly)
    2. Formats the prompt with or without retrieved articles
    3. Determines the data source category
    4. Creates the final data structure
    
    Args:
        example: Dictionary containing question data
        split: Dataset split ("train", "validation", or "test")
        idx: Index of the question in the dataset
        cutoff_date: Date cutoff for filtering questions
        
    Returns:
        Dictionary in the format expected by the training pipeline
    """
    # Determine number of articles to retrieve (randomly between 0-5)
    num_articles_to_retrieve = random.randint(0, 5)
    
    data_source = example.get("data_source", "Unknown")
    if not data_source:
        data_source = "unknown"
    data_source = data_source.lower()
    
    # Use all articles for metaculus since data is way in the past
    if "metaculus" in data_source.lower():
        num_articles_to_retrieve = 5
    
    # Extract question components
    question_raw = example.get("question_title", example.get("question", ""))
    background = example["background"]
    resolution_criteria = example["resolution_criteria"]
    if len(resolution_criteria) == 0:
        resolution_criteria = "N/A"
    answer = example["answer"]
    answer_type = example["answer_type"]
    resolution_date = example.get(
        "resolution_date",
        example.get("date_resolve_at", example.get("question_close_date", ""))
    )
    question_start_date = example.get("question_start_date", example.get("date_begin", ""))
    question_idx = example.get("question_idx", example.get("url", ""))
    relevant_docs_str = example.get("relevant_docs", [])
    
    # Parse the relevant_docs JSON string back to a list
    try:
        relevant_docs = json.loads(relevant_docs_str) if isinstance(relevant_docs_str, str) else relevant_docs_str
    except (json.JSONDecodeError, TypeError):
        relevant_docs = []
    
    # Format retrieved articles
    retrieved_news_articles_summaries = format_retrieved_articles(relevant_docs, num_articles_to_retrieve)
    
    # Determine data source category and format appropriate prompt
    news_source = example.get("news_source", "Unknown")
    if not news_source:
        news_source = "unknown"
    news_source = news_source.lower()
    
    resolution = int(example.get("resolution", -1))
    
    # Set default data field based on split
    if "train" in split:
        data_field = "freeform/cnn_dw_forbes_ht_irishtimes"
    else:
        data_field = "freeform/cnn_dw_forbes"
    
    # Format prompt based on question type
    if "manifold" in data_source:
        data_field = "binary/manifold"
        prompt = format_forecasting_prompt_binary_with_retrieval(
            question_raw, background, resolution_criteria, retrieved_news_articles_summaries
        )
        assert resolution == 1 or resolution == 0, "Resolution is not 1 or 0"
    elif "metaculus" in data_source:
        news_source = "binary/metaculus"
        data_field = "binary/metaculus"
        prompt = format_forecasting_prompt_binary_with_retrieval(
            question_raw, background, resolution_criteria, retrieved_news_articles_summaries
        )
        assert resolution == 1 or resolution == 0, "Resolution is not 1 or 0"
    elif "theguardian" in news_source:
        data_field = "freeform/theguardian-test"
        prompt = format_forecasting_prompt_with_retrieval(
            question_raw, background, resolution_criteria, answer_type, retrieved_news_articles_summaries
        )
    else:
        # Default: freeform question with retrieval
        prompt = format_forecasting_prompt_with_retrieval(
            question_raw, background, resolution_criteria, answer_type, retrieved_news_articles_summaries
        )
    
    # Create the final data structure
    data = {
        "data_source": data_field,
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "forecasting",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer_type": answer_type,
            "question_idx": question_idx,
            "answer": answer,
            "question": question_raw,
            "background": background,
            "resolution_criteria": resolution_criteria,
            "resolution_date": resolution_date,
            "question_source": data_field,
            "resolution": resolution,
            "prompt": prompt,
            "articles_retrieved": num_articles_to_retrieve,
            "with_retrieval": 1 if num_articles_to_retrieve > 0 else 0,
        },
    }
    
    return data


def save_dataset_to_jsonl(dataset: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the processed dataset to a JSONL file.
    
    Each line in the output file is a JSON object representing one example.
    This format is easy to load and view in text editors.
    
    Args:
        dataset: List of dictionaries, each representing one example
        output_path: Path where the JSONL file should be saved
    """
    logger.info(f"Saving dataset with {len(dataset)} examples to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Successfully saved dataset to {output_path}")


if __name__ == "__main__":
    print("Starting foresight_preparation.py...")
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(
        description="Preprocess forecasting dataset and save to JSONL format"
    )
    parser.add_argument(
        '--questions_file',
        type=str,
        default="/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_train_30.jsonl",
        help="Path to JSONL file containing articles with question fields"
    )
    parser.add_argument(
        "--local_dir",
        default="/fast/nchandak/forecasting/datasets/verl/freeform/data70k-retrieval/",
        help="Local directory to save the output JSONL file"
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the output to"
    )
    parser.add_argument(
        "--cutoff_date",
        default="2024-05-15",
        help="Cutoff date for filtering questions (YYYY-MM-DD format)"
    )

    args = parser.parse_args()
    print(f"Arguments: questions_file={args.questions_file}, local_dir={args.local_dir}, cutoff_date={args.cutoff_date}")
    sys.stdout.flush()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir, exist_ok=True)
        print(f"Created output directory: {args.local_dir}")
        sys.stdout.flush()
    
    # Determine dataset split from filename
    split = "train"
    if "validation" in args.questions_file:
        split = "validation"
    elif "test" in args.questions_file:
        split = "test"

    questions_data = []
    
    # Load questions based on split
    if "train" in args.questions_file:
        print(f"Loading training data from: {args.questions_file}")
        sys.stdout.flush()
        # Load main training questions
        questions_data = load_questions_from_jsonl(args.questions_file)
        print(f"Loaded {len(questions_data)} training questions")
        logger.info(f"Loaded {len(questions_data)} training questions")
        sys.stdout.flush()
        
        # Sort by resolution date
        questions_data.sort(key=lambda x: x["resolution_date"])
        
        # Load binary questions from Metaculus
        metaculus_train_data = load_questions_from_jsonl(
            "/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_metaculus_binary_train_2k_30.jsonl"
        )
        binary_data = metaculus_train_data
        binary_data.sort(key=lambda x: x["resolution_date"])
        
        # Filter questions after cutoff date and shuffle
        questions_data = [
            q for q in questions_data
            if is_date_after_cutoff(q["resolution_date"], args.cutoff_date)
        ]
        print(f"After cutoff date filter: {len(questions_data)} questions")
        logger.info(f"After cutoff date filter: {len(questions_data)} questions")
        sys.stdout.flush()
        
        # Shuffle the questions and binary data
        random.shuffle(questions_data)
        random.shuffle(binary_data)
        
        print(f"Before block size adjustment: {len(questions_data)} questions, {len(binary_data)} binary questions")
        logger.info(f"Before block size adjustment: {len(questions_data)} questions")
        logger.info(f"Before block size adjustment: {len(binary_data)} binary questions")
        sys.stdout.flush()
        
        # temporary split
        block_size = 256
        # Find the largest multiple of block_size that is less than or equal to the number of questions
        num_questions = len(questions_data)
        num_blocks = num_questions // block_size
        questions_data = questions_data[:num_blocks * block_size]
        
        num_binary_blocks = len(binary_data) // block_size
        binary_data = binary_data[:num_binary_blocks * block_size]
        
        print(f"After block size adjustment: {len(questions_data)} questions, {len(binary_data)} binary questions")
        logger.info(f"After block size adjustment: {len(questions_data)} questions")
        logger.info(f"After block size adjustment: {len(binary_data)} binary questions")
        sys.stdout.flush()
        
        # Combine with binary data
        questions_data.extend(binary_data)
        print(f"After adding binary data: {len(questions_data)} total questions")
        logger.info(f"After adding binary data: {len(questions_data)} total questions after block size adjustment")
        sys.stdout.flush()
        
    
    elif "validation" in args.questions_file or "test" in args.questions_file:
        # Load test/validation sets
        metaculus_test_data = load_questions_from_jsonl(
            "/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/with_retreival/metaculus-05-2025_30.jsonl"
        )
        theguardian_test_data = load_questions_from_jsonl(
            "/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_validation-theguardian_207_30.jsonl"
        )
        
        questions_data.extend(metaculus_test_data)
        questions_data.extend(theguardian_test_data)
        logger.info(f"Loaded {len(questions_data)} test/validation questions")
    
    # Process all questions into the final format
    print(f"Processing {len(questions_data)} questions...")
    logger.info("Processing questions...")
    sys.stdout.flush()
    processed_dataset = [
        process_question(example, split, idx, args.cutoff_date)
        for idx, example in enumerate(questions_data)
    ]
    
    print(f"Processed {len(processed_dataset)} examples")
    logger.info(f"Processed {len(processed_dataset)} examples")
    sys.stdout.flush()
    
    # Save to JSONL format
    input_filename = os.path.basename(args.questions_file)
    
    save_dir = args.local_dir.replace("data70k-retrieval", "foresight")
    output_path = os.path.join(args.local_dir, input_filename)
    output_path = output_path.replace(
        ".jsonl",
        f"_with_metaculus_randomk_shuffled_after_{args.cutoff_date}.jsonl"
    )
    
    print(f"Saving dataset to: {output_path}")
    sys.stdout.flush()
    save_dataset_to_jsonl(processed_dataset, output_path)
    print("Dataset saved successfully!")
    sys.stdout.flush()
    
    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        logger.info(f"Copying to HDFS: {args.hdfs_dir}")
        sys.stdout.flush()
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print("HDFS copy completed!")
        sys.stdout.flush()
    
    print("Script completed successfully!")
    sys.stdout.flush()
