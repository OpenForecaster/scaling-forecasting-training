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
Preprocess our own forecasting dataset to parquet format
"""

import argparse
import logging
import os
import re
import json
from typing import List
import numpy as np
import datasets
from datetime import datetime
from verl.utils.hdfs_io import copy, makedirs
import random

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



def format_forecasting_prompt_binary(
    question_title: str,
    background: str,
    resolution_criteria: str,
) -> str:
    """
    Format the prompt without article context.
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
    Check if date is after the cutoff date.
    
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
    """Format the prompt for single outcome forecasting."""
    
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



def format_forecasting_prompt_binary_with_retrieval(
    question_title: str,
    background: str,
    resolution_criteria: str,
    retrieved_news_articles_summaries: str,
) -> str:
    """Format the prompt for single outcome forecasting."""
    
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
    """Load articles with questions from JSONL file and extract question components."""
    questions_data = []
    print(f"Loading questions from {file_path}")
    
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
                    
                    resolution = article.get('resolution', -1)
                    qanswer = article.get('answer', '')
                    if "yes" in qanswer.lower():
                        resolution = 1
                    elif "no" in qanswer.lower():
                        resolution = 0
                    else :
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
                    
                    
                    # if date is in ISO format, convert to YYYY-MM-DD
                    if 'resolution_date' in question_entry:
                        if isinstance(question_entry['resolution_date'], int):
                            question_entry["resolution_date"] = datetime.fromtimestamp(question_entry['resolution_date']).strftime('%Y-%m-%d')
                    #     else :
                    #         print("NO resolution date:", question_entry['resolution_date'])
                    # else :
                    #     assert False, "No resolution date"
                            
                    if 'question_start_date' in question_entry:
                        if isinstance(question_entry['question_start_date'], int):
                            question_entry['question_start_date'] = datetime.fromtimestamp(question_entry['question_start_date']).strftime('%Y-%m-%d')
                        
                    # Only add if we have a valid question title
                    if question_entry['question_title'].strip():
                        questions_data.append(question_entry)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(questions_data)} valid questions from {file_path}")
    return questions_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_train_30.jsonl",
                       help="Path to JSONL file containing articles with question fields")
    parser.add_argument("--local_dir", default="/fast/nchandak/forecasting/datasets/verl/freeform/data70k-retrieval/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=100000)
    
    args = parser.parse_args()
    
    # Make local_dir if it doesn't exist
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir, exist_ok=True)
    
    split = "train"
    if "validation" in args.questions_file:
        split = "validation"
    if "test" in args.questions_file:
        split = "test"

    questions_data = []
    if "train" in args.questions_file:
        questions_data = load_questions_from_jsonl(args.questions_file)
        print(f"Length of questions_data: {len(questions_data)}")
        # sort questions_data by start date
        questions_data.sort(key=lambda x: x["resolution_date"])
        
        
        
        # binary_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_binary_train_30.jsonl")
        
        metaculus_train_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_metaculus_binary_train_2k_30.jsonl")
        # manifold_train_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_manifold_binary_train_2k_30.jsonl")
        
        binary_data = metaculus_train_data #+ manifold_train_data
        binary_data.sort(key=lambda x: x["resolution_date"])
        
        cutoff_date = "2024-06-01"
        # print the number of binary questions after the cutoff date
        print(f'Number of binary questions after the cutoff date: {len([q for q in binary_data if is_date_after_cutoff(q["resolution_date"], cutoff_date)])}')
        
        # convert to hf format
        # binary_data = datasets.Dataset.from_list(binary_data)
        # binary_data = binary_data.sort("resolution_date")
        # print columns of questions_data
        
        # print(binary_data[0])
        # print(binary_data[-1])
        # print(binary_data.column_names)
        
        # convert binary_data to list of dicts
        # binary_data = binary_data.to_list()
        
        # randomly shuffle questions data 
        random.shuffle(questions_data)
        random.shuffle(binary_data)
        
        # combine questions_data and binary_data
        # questions_data.extend(binary_data)
        print(f"Length of questions_data after extending with binary_data: {len(questions_data)}")
        # only keep those with resolution date after 2024-01-01
        questions_data = [q for q in questions_data if is_date_after_cutoff(q["resolution_date"], cutoff_date)]
        print(f"Length of questions_data after cutoff: {len(questions_data)}")

        # restrict the number of samples
        questions_data = questions_data[:args.num_samples]
        
    if "validation" in args.questions_file or "test" in args.questions_file:
        # Also load metaculus test set, and theguardian test set
        # metaculus_test_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/binary_test.jsonl")
        metaculus_test_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/with_retreival/metaculus-05-2025_30.jsonl")
        
        
        theguardian_test_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/synthetic/freeform/datamix/cnn-2024_dw-2024_forbes-2023_forbes-2024_hindustantimes-2024-25_irishtimes-2024/with_retrieval/ranked_queries_validation-theguardian_207_30.jsonl")
        
        print(f"Length of questions_data: {len(questions_data)}")
        questions_data.extend(metaculus_test_data)
        print(len(metaculus_test_data))
        print(f"Length of questions_data: {len(questions_data)}")
        questions_data.extend(theguardian_test_data)
        print(len(theguardian_test_data))
        print(f"Length of questions_data: {len(questions_data)}")
        
    # else :
    #     metaculus_train_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/synthetic/freeform/binary_train_2k.jsonl")
    #     questions_data.extend(metaculus_train_data)
    original_len = len(questions_data)
    check_metaculus = []
    check_manifold = []
    theguardian_test_data = []
    # Convert list of dicts to a HuggingFace Dataset object
    # dataset = datasets.Dataset.from_list(questions_data)

    # add a row to each data item that represents a unique id
    summary_lengths = []
    def make_map_fn(split, example, idx):
        num_articles_to_retrieve = random.randint(0, 5)
        # num_articles_to_retrieve = random.choice([0, 1, 2, 3, 4, 5])
        # print(f"num_articles_to_retrieve: {num_articles_to_retrieve}")
        
        # num_articles_to_retrieve = 5
        
        data_source = example.get("data_source", "Unknown")
        if not data_source:
            data_source = "unknown"
        data_source = data_source.lower()
        
        # Use all articles for metaculus since data is way in the past so retrieved articles is often < 5
        if "metaculus" in data_source.lower():
            num_articles_to_retrieve = 5
        
        question_raw = example.get("question_title", example.get("question", ""))
        background = example["background"]
        resolution_criteria = example["resolution_criteria"]
        if len(resolution_criteria) == 0:
            resolution_criteria = "N/A"
        answer = example["answer"]
        answer_type = example["answer_type"]
        resolution_date = example.get("resolution_date", example.get("date_resolve_at", example.get("question_close_date", "")))
        question_start_date = example.get("question_start_date", example.get("date_begin", ""))
        question_idx = example.get("question_idx", example.get("url", ""))
        relevant_docs_str = example.get("relevant_docs", [])
        
        # Parse the relevant_docs JSON string back to a list
        try:
            relevant_docs = json.loads(relevant_docs_str) if isinstance(relevant_docs_str, str) else relevant_docs_str
        except (json.JSONDecodeError, TypeError):
            relevant_docs = []
        
        # Format the prompt for each example
        retrieved_news_articles_summaries = ""
        
        j = 1 
        for doc in relevant_docs:
            if j > num_articles_to_retrieve:
                break
            
            article_title = None
            article_summary = None
            article_passage = None
            article_source = None
            source_text = ""
            date_text = ""
            author_text = ""
            # if not isinstance(doc[2], dict) and isinstance(doc[2], str):
                
            for item in doc:
                if isinstance(item, dict):
                    if "title" in item:
                        article_title = item["title"]
                        
                    if "relevant_passage" in item:
                        article_passage = item["relevant_passage"]
                        
                    elif "summary" in item and item["prompt_name"] == "create_forecast_summarization_prompt":
                        article_summary = item["summary"]
                        
                    if "source_domain" in item:
                        article_source = item["source_domain"]
                        source_text = f"Source: {article_source}\n"
                            
                    if "max_date" in item:
                        article_date = item["max_date"]
                        # this is in ISO format in int, convert to human readable format
                        article_date = datetime.fromtimestamp(article_date).strftime("%B %d, %Y") 
                        date_text = f"Article Publish Date: {article_date}\n"
                        
                    if "authors" in item and len(item["authors"]) > 0:
                        first5authors = item["authors"][:5]
                        first5 = ",".join(first5authors)
                        author_text = f"Article Author(s): {first5}\n"
                        
                if article_title is not None :
                    if article_passage is not None:
                        retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\n{source_text}{author_text}{date_text}Relevant Passage: {article_passage}\n\n"
                    elif article_summary is not None:
                        retrieved_news_articles_summaries += f"Article {j}:\nTitle: {article_title}\n{source_text}{author_text}{date_text}Summary: {article_summary}\n\n"
                        
            j += 1



        prompt = format_forecasting_prompt_with_retrieval(question_raw, background, resolution_criteria, answer_type, retrieved_news_articles_summaries)

        
        news_source = example.get("news_source", "Unknown")
        if not news_source:
            news_source = "unknown"
        news_source = news_source.lower()
        
        resolution = int(example.get("resolution", -1))
        
        # prompt = format_forecasting_prompt(question_raw, background, resolution_criteria, answer_type)

        
        if "train" in split:
            data_field = "freeform/cnn_dw_forbes_ht_irishtimes"
        else :
            data_field = "freeform/cnn_dw_forbes"
        
        if "manifold" in data_source:
            data_field = "binary/manifold"
            prompt = format_forecasting_prompt_binary_with_retrieval(question_raw, background, resolution_criteria, retrieved_news_articles_summaries)
            check_manifold.append(prompt)
            assert resolution == 1 or resolution == 0, "Resolution is not 1 or 0"
            
        elif "metaculus" in data_source:
            news_source = "binary/metaculus"
            data_field = "binary/metaculus"
            # prompt = format_forecasting_prompt_binary(question_raw, background, resolution_criteria)
            prompt = format_forecasting_prompt_binary_with_retrieval(question_raw, background, resolution_criteria, retrieved_news_articles_summaries)
            
            check_metaculus.append(prompt)
            assert resolution == 1 or resolution == 0, "Resolution is not 1 or 0"
        elif "theguardian" in news_source:
            data_field = "freeform/theguardian-test"
            theguardian_test_data.append(prompt)
        # else:
        #     data_field = "freeform/cnn_dw_forbes"

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
                # "response": f"<answer> {answer} </answer> <probability> 0.95 </probability>",
                "with_retrieval": 1 if num_articles_to_retrieve > 0 else 0,
            },
        }
        
        # if idx < 1 :
        #     print(data["prompt"][0]["content"])
        #     print("-"*100)
        #     print("-"*100)  
        #     print("\n\n")
            
            
        return data

    # processed_dataset = dataset.map(function=make_map_fn(split), with_indices=True)
    
    processed_dataset = []
    for j in range(1):
        cur_dataset = [make_map_fn(split, example, idx) for idx, example in enumerate(questions_data)]
        processed_dataset = processed_dataset + cur_dataset
    
    print(f"Length of the WHOLE dataset: {len(processed_dataset)}")
    
    processed_dataset = datasets.Dataset.from_list(processed_dataset)
    
    # print first 2 rows of processed_dataset
    print(processed_dataset[0]["prompt"])
    print(processed_dataset[-1]["prompt"])
    
    print(f"Length of processed dataset: {len(processed_dataset)}")
    
    if len(check_metaculus) > 0:
        print("-"*100)
        print("Metaculus prompts")
        print("-"*100)
        print(check_metaculus[0])
        print(check_metaculus[-1])
        
    if len(check_manifold) > 0:
        print("-"*100)
        print("Manifold prompts")
        print("-"*100)
        print(check_manifold[0])
        print(check_manifold[-1])

    # print min, max, mean, median, 75th percentile, 90th percentile, 95th percentile, 99th percentile of summary lengths
    # print(f"Min summary length: {min(summary_lengths)}")
    # print(f"Max summary length: {max(summary_lengths)}")
    # print(f"Mean summary length: {sum(summary_lengths) / len(summary_lengths)}")
    # print(f"Median summary length: {np.median(summary_lengths)}")
    # print(f"75th percentile summary length: {np.percentile(summary_lengths, 75)}")
    # print(f"90th percentile summary length: {np.percentile(summary_lengths, 90)}")
    # print(f"95th percentile summary length: {np.percentile(summary_lengths, 95)}")
    # print(f"99.9th percentile summary length: {np.percentile(summary_lengths, 99.9)}")
    # print(f"99.99th percentile summary length: {np.percentile(summary_lengths, 99.99)}")
    
    # Print every K occurrence of the first question
    
    which = 22
    idx = which
    print(original_len, which, processed_dataset[idx]["extra_info"]["prompt"], "Articles retrieved:",  processed_dataset[idx]["extra_info"]["articles_retrieved"], "\n\n")
    
    
    # idx = original_len*2 + which
    # print(original_len*2, which, processed_dataset[idx]["extra_info"]["prompt"], "Articles retrieved:",  processed_dataset[idx]["extra_info"]["articles_retrieved"], "\n\n")
    
    # idx = original_len*3 + which
    # print(original_len*3, which, processed_dataset[idx]["extra_info"]["prompt"], "Articles retrieved:",  processed_dataset[idx]["extra_info"]["articles_retrieved"], "\n\n")
    
    
    # idx = original_len*4 + which
    # print(original_len*4, which, processed_dataset[idx]["extra_info"]["prompt"], "Articles retrieved:",  processed_dataset[idx]["extra_info"]["articles_retrieved"], "\n\n")
    
    # print(original_len*2, which, processed_dataset[original_len*2 + which]["extra_info"]["prompt"], "\n\n")
    # print(original_len*4, which, processed_dataset[original_len*4 + which]["extra_info"]["prompt"], "\n\n")
    # print(original_len*3, which, processed_dataset[original_len*3 + which]["prompt"], "\n\n")
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save with same name as input file
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
    input_filename = os.path.basename(args.questions_file)
    output_path = os.path.join(local_dir, input_filename)
    # output_path = output_path.replace(".jsonl", "_with_metaculus_randomk_shuffled.jsonl")
    output_path = output_path.replace(".jsonl", f"_randomk_{args.num_samples}_shuffled.jsonl")
    processed_dataset.to_parquet(output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
