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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/fast/nchandak/forecasting/misc/polaris")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "POLARIS-Project/Polaris-Dataset-53K"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    dataset = dataset["train"]
    
    
    # Train should be the one with column `difficulty == '2/8'` and test should be the one with column `difficulty == '0/8'`
    train_dataset = dataset.filter(lambda x: x["difficulty"] == "1/8") #  or x["difficulty"] == "2/8")
    test_dataset = dataset.filter(lambda x: x["difficulty"] == "0/8")
    
    # only kee 4000 random samples from test set
    train_dataset = train_dataset.select(np.random.choice(len(train_dataset), 4000, replace=False))
    # only keep 1000 random samples from test set
    test_dataset = test_dataset.select(np.random.choice(len(test_dataset), 1000, replace=False))

    instruction_following = "You will be asked a math question. Think step by step and output your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = instruction_following + "\nProblem: " + question 

            answer = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train4k.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
        
    # example = test_dataset[0]
    # with open(os.path.join(local_dir, "test_example.json"), "w") as f:
    #     json.dump(example, f, indent=2)
        
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
