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


def load_results_from_jsonl(file_path: str) -> List[dict]:
    """Load articles with questions from JSONL file and extract question components."""
    results_data = []
    print(f"Loading results from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    result = json.loads(line.strip())
                    results_data.append(result)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_idx}: {e}")
                    continue
    
    logger.info(f"Loaded {len(results_data)} valid results from {file_path}")
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions_file', type=str, default="/fast/nchandak/forecasting/evals/freeform/distillation/theguardian_9735/grok-3-mini_eval_size_9735_generations_1.jsonl",
                       help="Path to JSONL file containing articles with question fields")
    parser.add_argument("--local_dir", default="/fast/nchandak/forecasting/datasets/verl/freeform/distillation/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    
    # Make local_dir if it doesn't exist
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir, exist_ok=True)
    
    split = "train"
    if "validation" in args.questions_file:
        split = "validation"
    if "test" in args.questions_file:
        split = "test"

    results_data = load_results_from_jsonl(args.questions_file)
    print(f"Length of results_data: {len(results_data)}")
    
    # Randomly shuffle and split into train and validation
    random.shuffle(results_data)
    train_data = results_data[:int(0.95 * len(results_data))]
    validation_data = results_data[int(0.95 * len(results_data)):]
    
    # add a row to each data item that represents a unique id
    summary_lengths = []
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.get("question_title", example.get("question", ""))
            background = example["background"]
            resolution_criteria = example["resolution_criteria"]
            answer = example["answer"]
            answer_type = example["answer_type"]
            resolution_date = example.get("resolution_date", example.get("date_resolve_at", example.get("question_close_date", "")))
            question_start_date = example.get("question_start_date", example.get("date_begin", ""))
            question_idx = example.get("question_idx", example.get("idx", ""))
            reasoning = example["reasoning"][0]
            response = example["response"][0]
            prompt = example["prompt"][0]
            full_response = f"<think> {reasoning} </think> {response}"

            # prompt = format_forecasting_prompt_with_retrieval(question_raw, background, resolution_criteria, answer_type, retrieved_news_articles_summaries)

            data_source = example.get("data_source", "Unknown").lower()
            news_source = example.get("news_source", "Unknown").lower()
            resolution = int(example.get("resolution", -1))
            data_field = "freeform/distillation-grok-3-mini"

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
                    "response": full_response,
                    "prompt": prompt,
                    "with_retrieval": True,
                },
            }
            
            # if idx < 1 :
            #     print(data["prompt"][0]["content"])
            #     print("-"*100)
            #     print("-"*100)  
            #     print("\n\n")
                
                
            return data

        return process_fn

    # processed_dataset = dataset.map(function=make_map_fn(split), with_indices=True)
    train_dataset = [make_map_fn(split)(example, idx) for idx, example in enumerate(train_data)]
    train_dataset = datasets.Dataset.from_list(train_dataset)
    
    # print first 2 rows of processed_dataset
    # print(train_dataset[0])
    # print(train_dataset[-1])
    print(f"Length of processed train dataset: {len(train_dataset)}")

    validation_dataset = [make_map_fn(split)(example, idx) for idx, example in enumerate(validation_data)]
    validation_dataset = datasets.Dataset.from_list(validation_dataset)
    # print(validation_dataset[0])
    # print(validation_dataset[-1])
    print(f"Length of processed validation dataset: {len(validation_dataset)}")
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save with same name as input file
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
    input_filename = os.path.basename(args.questions_file)
    train_output_path = os.path.join(local_dir, input_filename.replace(".jsonl", "_train.parquet"))
    validation_output_path = os.path.join(local_dir, input_filename.replace(".jsonl", "_validation.parquet"))
    print(f"Saving train dataset to {train_output_path}")
    print(f"Saving validation dataset to {validation_output_path}")
    train_dataset.to_parquet(train_output_path)
    validation_dataset.to_parquet(validation_output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
        copy(src=train_output_path, dst=hdfs_dir)
        copy(src=validation_output_path, dst=hdfs_dir)
