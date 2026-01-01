#!/usr/bin/env python3
"""
Infinite Games Binary Questions Analyzer

Purpose:
    Analyzes binary forecasting questions from Infinite Games platform.
    Filters to LLM market type and generates training/test datasets.

Main Operations:
    - Load CSV data from Infinite Games
    - Filter to LLM market type questions
    - Convert to binary format (0/1 resolutions)
    - Generate statistics and visualizations
    - Export to JSONL format for model training/evaluation

Data Source:
    Infinite Games platform - past events CSV export

Usage:
    python analyze_binary.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Load the CSV data
df = pd.read_csv("/fast/nchandak/forecasting/datasets/infinitegames/past_1000_events_2025-03-18T1642.csv")

# Add idx column
df['idx'] = df.index

# Print all the columns names 
print(df.columns)

# Print df length
print("OG length:", len(df))
# Create new column resolution based on outcome column
# df['resolution'] = df['outcome'].apply(lambda x: 1 if isinstance(x, str) and "yes" in x.lower() else 0)
df['resolution'] = df['outcome'].apply(lambda x: 1 if x else 0)

# Only keep rows with market_type == "llm"
df = df[df['market__type'] == "llm"]

# Print top 10 rows
print(df.head(10))
print(df.tail(10))

print("New length:", len(df))

# df['prefix'] = df['question'].apply(lambda x: x[:5])

# print(df['prefix'].value_counts())

# Count the number of question which don't have a year in it 
# print("Questions without year:", len(df[~df['question'].str.contains(r'\d{4}')]))
# # Remove these rows
# df = df[df['question'].str.contains(r'\d{4}')]

# print resolution value counts
print(df['resolution'].value_counts())


# iterate over all rows and print the question, answer, resolution, and total_points
idx = 0
rows_to_keep = []

for index, row in df.iterrows():
    metadata = row['event_metadata']
    # convert str to dict
    metadata = json.loads(metadata)
    # print(metadata)
    
    ans = metadata.get("resolver", {}).get('answer', None)
    
    if ans is None:
        continue
    
    if not "yes" in ans.lower() and not "no" in ans.lower():
        continue
    
    perplexity = metadata.get("resolver", {}).get("perplexity", None)
    
    if perplexity is None:
        continue 
    
    resolution = perplexity.get("resolution", None)
    
    if resolution is None:
        continue
    
    if not "yes" in resolution.lower() and not "no" in resolution.lower():
        continue
    
    resolver = metadata.get("resolver", {})
    
    # if 'serper' in resolver:
        # print the title, description, and resolution
        # print(f"Title: {row['title']}")
        # print(f"Description: {row['description']}")
        # print(f"Resolution: {resolution}")
        # print("-"*100)
    
    rows_to_keep.append(index)
    # print(metadata)
    
    # if perplexity is not None:
    #     resolution = perplexity.get("resolution", None)
        
    idx += 1
    # if idx > 50:
    #     break
    
print(len(rows_to_keep))

# filter the df
df = df.loc[rows_to_keep]

# print resolution value counts
print(df['resolution'].value_counts())

idx = 0
# print metadata of first few rows 
for index, row in df.iterrows():
    metadata = row['event_metadata']
    # convert str to dict
    metadata = json.loads(metadata)
    # print(metadata)
    idx += 1
    if idx > 2500:
        break


def format_forecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    zero_shot: bool = True
) -> str:
    """
    Format the prompt given the row data.
    """
    if zero_shot:
        return f"""You will be asked a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. 

Question: {question}
Resolution Criteria: {resolution_criteria}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
"""
    else:
        # If not zero_shot, you can modify the prompt as needed.
        return f"""
Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}
"""

# Create dataset for huggingface
import pandas as pd

# Randomly sample 1000 rows from df_final
# df_final = df_final.sample(n=1000)

data_list =[]

for idx, row in df.iterrows():
    # Use title as the question, and body as the background.
    question = row["title"]
    background = "Not available"
    
    resolution_criteria = row["description"]
    # Convert resolution to binary (1 for yes, 0 for no)
    resolution = row["resolution"]
    # Convert date to string format to avoid JSON serialization issues with Timestamp objects
    date_resolve = row["resolved_at"]

    # Extract URLs from background if any exist
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row["description"])

    # Create dictionary for this example
    example_dict = {
        'date_resolve_at': date_resolve,
        'extracted_urls': urls,
        'question_type': "binary",
        # 'url': row["url"],
        'background': background, # row["body"],
        'resolution_criteria': resolution_criteria,
        'is_resolved': True,
        'date_close': date_resolve,
        'question': question,
        'data_source': "infinitegames_binary",
        'resolution': resolution,
        "id": row["event_id"],
        "external_id": row["external_id"],
        'idx': row["idx"],
    }

    # Create the prompt (change zero_shot to True if desired)
    prompt = format_forecasting_prompt(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=str(date_resolve),
        date_close=str(date_resolve),
        zero_shot=False,
    )
    
    # Append the prompt and resolution together.
    combined_output = prompt 
    example_dict['prompt'] = prompt
    example_dict['full_prompt'] = format_forecasting_prompt(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=str(date_resolve),
        date_close=str(date_resolve),
        zero_shot=True,
    )
    
    data_list.append(example_dict)

import random
random.shuffle(data_list)
# data_list = data_list[:4]
print(len(data_list))
# Save data_list in proper format
suffix = "test"

# Create a balanced subset of data_list based on resolution (try to maximize the number of examples)
data_list_1 = [example for example in data_list if example['resolution'] == 1]
data_list_0 = [example for example in data_list if example['resolution'] == 0]

min_sample_size = min(len(data_list_1), len(data_list_0))
# Randomly sample 1000 rows from data_list_1 and data_list_0
data_list_1 = random.sample(data_list_1, min_sample_size)
data_list_0 = random.sample(data_list_0, min_sample_size)

data_list2 = data_list_1 + data_list_0

# shuffle data_list2
random.shuffle(data_list2)
random.shuffle(data_list)

print("Length of balanced subset:", len(data_list2))

# print(f"Example prompts of {suffix} set:\n\n")

random_sample = random.sample(data_list2, 10)
for example in random_sample:
    # print(example['prompt'])
    print("Date resolve at:", example['date_resolve_at'])
    print(example['prompt'])
    print("\nResolution:", example['resolution'])
    # print(example['idx'])
    print("-"*100)
    
file_path = f"/fast/nchandak/forecasting/datasets/infinitegames/binary_balanced_test.json"
# print(f"Saving to {file_path} with data_list length {len(data_list)}")
with open(file_path, 'w') as f:
    print(f"Saving to {file_path} with data_list length {len(data_list2)}")
    json.dump(data_list2, f, indent=4, ensure_ascii=False)
    
file_path = f"/fast/nchandak/forecasting/datasets/infinitegames/binary_test.json"
# print(f"Saving to {file_path} with data_list length {len(data_list)}")
with open(file_path, 'w') as f:
    print(f"Saving to {file_path} with data_list length {len(data_list)}")
    json.dump(data_list, f, indent=4, ensure_ascii=False)

# # Only keep a random 1000 rows
# # import random
# # random.shuffle(data_list)
# # data_list = data_list[:1000]
