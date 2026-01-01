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
import random
import datasets

from verl.utils.hdfs_io import copy, makedirs

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
    question_start_date: str,
) -> str:
    """
    Format the prompt without article context.
    """
    
    prompt = f"""You will be asked a BINARY forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your probability estimate of it resolving YES).
        
Question Title: {question_title}
Question Start Date: {question_start_date}
Question Background: {background}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final probability for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""

    return prompt

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

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically (- (1 - p)^2) if your answer is correct and (- 1 - p^2) if your answer is incorrect. For example, if p = 0.5, and your answer is incorrect, then your score will be (-1 - 0.5^2) = (-1 - 0.25) = -1.25 whereas if the answer was correct, then your score would be (- (1 - 0.5)^2) = (- (0.5)^2) = -0.25. Thus, the range of the score is [-2, 0] where your score lies between [-2, -1] if the answer is incorrect and [-1, 0] if the answer is correct. If your answer is correct, your will be REWARDED more if your probability is higher whereas if your answer is incorrect, your will be PENALIZED more if your probability is higher. TRY HARD TO COME UP WITH THE BEST GUESS FOR THE FINAL ANSWER. YOU HAVE TO MAXIMIZE YOUR SCORE.

Your final answer should be concise (NOT MORE THAN A FEW WORDS LONG) and your response SHOULD STRICTLY END with <answer> </answer> tags and <probability> </probability> tags."""

    return prompt



def load_manifold_and_metaculus_data(split="train", vol_filter_limit = 4000):
    path1 = "/fast/nchandak/forecasting/datasets/manifold/manifold_binary_train_with_r1queries.json"
    path2 = "/fast/nchandak/forecasting/datasets/metaculus/old-2024/binary_"
    
    path2 += "raw_"
        
    path2 += split + ".json"
    
    
    # Load dataset using Dataset.from_json() instead of load_dataset()
    ds1 = datasets.Dataset.from_json(path1)
    ds2 = datasets.Dataset.from_json(path2)
    
    # print length of individual datasets
    print("Length of dataset 1: ", len(ds1))
    print("Length of dataset 2: ", len(ds2))
    
    # filter ds1 based on volume
    ds1 = ds1.filter(lambda x: x["volume"] >= vol_filter_limit)
    print("Length of dataset 1 after filtering: ", len(ds1))
    
    # only keep columns which are present in both datasets
    to_remove = []
    for col in ds1.column_names:
        if col not in ds2.column_names:
            to_remove.append(col)
            
    ds1 = ds1.remove_columns(to_remove)
    
    to_remove = []
    for col in ds2.column_names:
        if col not in ds1.column_names:
            to_remove.append(col)
            
    ds2 = ds2.remove_columns(to_remove)
    
    # Concatenate the two datasets
    # ds = datasets.concatenate_datasets([ds1, ds2])
    ds = ds1 
    print("Length of concatenated dataset: ", len(ds))
    
    # Print the row with the longest prompt length
    # max_length = 0
    # max_row = None
    # for row in ds:
    #     length = len(row["question"]) + len(row["background"]) + len(row["resolution_criteria"])
    #     if length > max_length:
    #         max_length = length
    #         max_row = row
    
    # print("Max row: ", max_row)
    # print("Max length: ", max_length)
    
    # Remove all rows where prompt length (question + background + resolution criteria) is more than 20000 characters
    ds = ds.filter(lambda x: len(x["question"]) + len(x["background"]) + len(x["resolution_criteria"]) <= 10000)
    print("Length of dataset after removing rows with prompt length > 10000: ", len(ds))
    
    # print("Length of dataset after removing duplicates: ", len(ds))
    return ds
    


def load_questions_from_jsonl(file_path: str) -> List[dict]:
    """Load articles with questions from JSONL file and extract question components."""
    questions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    article = json.loads(line.strip())
                    
                    # Skip if question is not relevant or the article is not relevant
                    # if 'question_relevant' in article and int(article['question_relevant']) == 0:
                    #     continue
                    
                    # if 'article_relevant' in article and int(article['article_relevant']) == 0:
                    #     continue
                    
                    if 'no_good_question' in article and int(article['no_good_question']) == 1:
                        continue
                    
                    # Create a question entry with all necessary fields
                    question_entry = {
                        'idx': line_idx,
                        'question_title': article.get('question_title', article.get('question', '')),
                        'background': article.get('background', ''),
                        'resolution_criteria': article.get('resolution_criteria', ''),
                        'answer': article.get('answer', ''),
                        'answer_type': article.get('answer_type', ''),
                        'resolution_date': article.get('resolution_date', ''),
                        'question_start_date': article.get('question_start_date', article.get('date_begin', '')),
                        'question_idx': article.get('question_idx', line_idx),
                        'url': article.get('url', ''),
                        'data_source': article.get('data_source', ''),
                        'news_source': article.get('news_source', ''),
                        'resolution': int(article.get('resolution', -1)),
                    }
                    
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
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/synthetic/freeform/forbes24/filtered_train_10k.jsonl",
                       help="Path to JSONL file containing articles with question fields")
    parser.add_argument("--local_dir", default="/fast/nchandak/forecasting/datasets/verl/binary-ablation/both/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    
    split = "train"
    if "validation" in args.questions_file:
        split = "validation"
    if "test" in args.questions_file:
        split = "test"

    questions_data = load_questions_from_jsonl(args.questions_file)

    if "train" in args.questions_file:
        binary_data = load_manifold_and_metaculus_data(split="train", vol_filter_limit=4000)
        binary_data = binary_data.select(range(10000))
        # print columns of questions_data
        print(binary_data.column_names)
        
        # convert binary_data to list of dicts
        binary_data = binary_data.to_list()
        
        # combine questions_data and binary_data
        questions_data.extend(binary_data)
        
        # randomly shuffle questions_data which is a list of dicts with a specific seed 
        random.seed(42)
        random.shuffle(questions_data)
        print("length of questions_data: ", len(questions_data))
    
    if "validation" in args.questions_file or "test" in args.questions_file:
        # Also load metaculus test set, and theguardian test set
        metaculus_test_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/metaculus/fromMay2025/binary_test.jsonl")
        theguardian_test_data = load_questions_from_jsonl("/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian_207_free_3_cleaned.jsonl")
        
        questions_data.extend(metaculus_test_data)
        # print(len(metaculus_test_data))
        questions_data.extend(theguardian_test_data)
        # print(len(theguardian_test_data))
        
    # else :
    #     metaculus_train_data = load_questions_from_jsonl("/fast/nchandak/forecasting/datasets/synthetic/freeform/binary_train_2k.jsonl")
    #     questions_data.extend(metaculus_train_data)
        
        

    # Convert list of dicts to a HuggingFace Dataset object
    check_metaculus = []
    num_binary = 0
    num_freeform = 0

    # Helper to safely cast to int, returns fallback if fails
    def safe_int(val, fallback=-1):
        try:
            if val is None:
                return fallback
            return int(val)
        except (ValueError, TypeError):
            return fallback

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            global num_binary, num_freeform
            question_raw = example.get("question_title", example.get("question", ""))
            background = example.get("background", "")
            resolution_criteria = example.get("resolution_criteria", "")

            # Ensure resolution is always an int or fallback
            resolution_val = example.get("resolution", -2)
            resolution = safe_int(resolution_val, -2)
            if resolution >= -0.1: # Resolution is 0 or 1
                answer = "YES" if resolution == 1 else "NO"
            else:
                answer = example.get("answer", "UNKNOWN")

            # answer_type should be a string
            answer_type = str(example.get("answer_type", "binary"))
            
            # if "no" == answer.lower():
            #     # if "binary" not in answer_type.lower():
            #     # print("answer: ", answer)
            #     print("answer_type: ", answer_type)
            #     print("question: ", example.get("question", ""))
            #     print("question_title: ", example.get("question_title", ""))
            #     # print("background: ", example.get("background", ""))
            #     # print("resolution_criteria: ", example.get("resolution_criteria", ""))
            #     # print("resolution_date: ", example.get("resolution_date", ""))
            #     # assert False 
            
            resolution_date = example.get(
                "resolution_date",
                example.get("date_resolve_at", example.get("question_close_date", ""))
            )
            question_start_date = example.get(
                "question_start_date",
                example.get("date_begin", "")
            )
            question_idx = example.get("question_idx", example.get("url", ""))

            data_source = example.get("data_source", "manifold")
            if data_source is None:
                data_source = "manifold"
            data_source = str(data_source).lower()

            if "binary" in answer_type.lower():
                num_binary += 1
            else:
                num_freeform += 1

            news_source = example.get("news_source", "Unknown")
            if news_source is None:
                news_source = "Unknown"
            news_source = str(news_source).lower()

            # Ensure resolution is always an int or fallback
            resolution = safe_int(example.get("resolution", -1), -1)

            prompt = format_forecasting_prompt(question_raw, background, resolution_criteria, answer_type)

            data_field = "freeform/cnn_dw_forbes"
            if "manifold" in data_source:
                data_field = "binary/manifold-train"
                prompt = format_forecasting_prompt_binary(question_raw, background, resolution_criteria, question_start_date)
            elif "metaculus" in data_source:
                data_field = "binary/metaculus-test"
                prompt = format_forecasting_prompt_binary(question_raw, background, resolution_criteria, question_start_date)
                check_metaculus.append(example)
            elif "theguardian" in news_source:
                data_field = "freeform/theguardian-test"
            elif "forbes24" in data_source:
                data_field = "freeform/forbes24"
            else:
                data_field = "freeform/cnn_dw_forbes"

            # Ensure all fields in extra_info are of correct type for Arrow
            # Especially: index (int), resolution (int), question_idx (str)
            # Also, avoid any accidental non-string or non-int types

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
                    "split": str(split),
                    "index": int(idx),
                    "answer_type": str(answer_type),
                    "question_idx": str(question_idx),
                    "answer": str(answer),
                    "question": str(question_raw),
                    "background": str(background),
                    "resolution_criteria": str(resolution_criteria),
                    "resolution_date": str(resolution_date),
                    "question_source": str(data_field),
                    "resolution": int(resolution),
                },
            }
            return data
        return process_fn

    # 'questions_data' is a list, so use list comprehension instead of .map
    processed_dataset = [make_map_fn(split)(example, idx) for idx, example in enumerate(questions_data)]
    
    # sort the list by resolution date
    # processed_dataset.sort(key=lambda x: x["extra_info"]["resolution_date"])
    
    processed_dataset = datasets.Dataset.from_list(processed_dataset)
    
    # print first 2 rows of processed_dataset
    # print(processed_dataset[0])
    # print(processed_dataset[1])
    # print(processed_dataset[2])
    # print(processed_dataset[3])
    print(processed_dataset[-2])
    print(processed_dataset[-1])
    print("num_binary: ", num_binary)
    print("num_freeform: ", num_freeform)
    # print(check_metaculus[0])
    # print(check_metaculus[-1])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save with same name as input file
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
    input_filename = "both_filtered_train_20k.jsonl" # os.path.basename(args.questions_file)
    output_path = os.path.join(local_dir, input_filename)
    processed_dataset.to_parquet(output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
