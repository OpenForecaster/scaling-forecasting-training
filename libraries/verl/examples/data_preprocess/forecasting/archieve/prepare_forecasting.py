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
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/datasets/synthetic/freeform/cnn_dw_forbes/combined_all_questions_non_numeric_clean_train.jsonl",
                       help="Path to JSONL file containing articles with question fields")
    parser.add_argument("--local_dir", default="/fast/nchandak/forecasting/datasets/verl/freeform/forbes23_24/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    
    split = "train"
    if "validation" in args.questions_file:
        split = "validation"
    if "test" in args.questions_file:
        split = "test"

    questions_data = load_questions_from_jsonl(args.questions_file)
    # questions_data.sort(key=lambda x: x["question_start_date"])
    questions_data.sort(key=lambda x: x["resolution_date"])
    
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
    dataset = datasets.Dataset.from_list(questions_data)
    check_metaculus = []
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.get("question_title", example.get("question", ""))
            background = example["background"]
            resolution_criteria = example["resolution_criteria"]
            answer = example["answer"]
            answer_type = example["answer_type"]
            resolution_date = example.get("resolution_date", example.get("date_resolve_at", example.get("question_close_date", "")))
            question_start_date = example.get("question_start_date", example.get("date_begin", ""))
            question_idx = example.get("question_idx", example.get("url", ""))
            
            data_source = example.get("data_source", "Unknown").lower()
            news_source = example.get("news_source", "Unknown").lower()
            resolution = int(example.get("resolution", -1))
            
            prompt = format_forecasting_prompt(question_raw, background, resolution_criteria, answer_type)

            data_field = "freeform/cnn_dw_forbes"
            if "metaculus" in data_source:
                data_field = "binary/metaculus-test"
                prompt = format_forecasting_prompt_binary(question_raw, background, resolution_criteria, question_start_date)
                check_metaculus.append(example)
                
            elif "theguardian" in news_source:
                data_field = "freeform/theguardian-test"
            elif "forbes24" in data_source:
                data_field = "freeform/forbes24"
            else:
                data_field = "freeform/cnn_dw_forbes"

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
                },
            }
            return data

        return process_fn

    processed_dataset = dataset.map(function=make_map_fn(split), with_indices=True)
    
    # print first 2 rows of processed_dataset
    print(processed_dataset[0])
    # print(processed_dataset[1])
    print(processed_dataset[-2000])
    print(processed_dataset[-1])
    
    # print(check_metaculus[0])
    # print(check_metaculus[-1])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save with same name as input file
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
    input_filename = os.path.basename(args.questions_file)
    output_path = os.path.join(local_dir, input_filename)
    processed_dataset.to_parquet(output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
