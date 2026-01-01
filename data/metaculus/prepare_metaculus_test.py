#!/usr/bin/env python3
"""
Metaculus Binary Test Set Preparation

Purpose:
    Creates test/train datasets from Metaculus binary questions with proper formatting
    and quality filtering. Generates prompts for binary forecasting evaluation.

Main Steps:
    1. Load and filter Metaculus questions (binary type, resolved yes/no)
    2. Filter by date range and number of forecasters
    3. Remove practice questions and low-quality questions
    4. Generate standardized forecasting prompts
    5. Export to JSONL format for evaluation

Output Fields:
    - question: Question title
    - background: Question description/context
    - resolution_criteria: How the question will be resolved
    - prompt: Formatted prompt for LLM evaluation
    - resolution: Binary answer (0=NO, 1=YES)
    - answer_type: "binary (yes/no)"
    - date fields: date_begin, date_close, date_resolve

Usage:
    python prepare_metaculus_test.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load data from .jsonl file
jsonl_path = "/fast/nchandak/forecasting/datasets/metaculus/fromOct2025/metaculusFromOct.jsonl"
with open(jsonl_path, "r") as f:
    data = [json.loads(line) for line in f]

# Convert the data into a DataFrame
df = pd.DataFrame(data)


# Convert 'created_date' to datetime (timezone-aware)
df['created_date'] = pd.to_datetime(df['created_date'])

# Create timezone-aware datetime objects for filtering
start_date = pd.Timestamp("2024-07-01", tz='UTC')
end_date = pd.Timestamp("2024-12-31 23:59:59", tz='UTC')

# No end date
end_date = pd.Timestamp("2025-12-31 23:59:59", tz='UTC')

# Filter for binary questions and make a copy
df_binary = df[df['question_type'] == "binary"].copy()

# Filter for questions created within the specified date range and make a copy
# df_date_filtered = df_binary[(df_binary['created_date'] >= start_date) & 
#                              (df_binary['created_date'] <= end_date)].copy()

# # Use .loc for assignment: Extract 'nr_forecasters' from the metadata
# df_date_filtered.loc[:, 'nr_forecasters'] = df_date_filtered['metadata'].apply(lambda x: x.get('nr_forecasters', 0))

# # Filter for questions with more than 10 forecasters.
# df_final = df_date_filtered[df_date_filtered['nr_forecasters'] > 10].copy()

df_final = df_binary.copy()

# Print resolution value counts before filtering
print("\nResolution value counts before filtering:")
print(df_final['resolution'].value_counts())

# Filter for questions with resolution as yes or no (case-insensitive)
df_final = df_final[df_final['resolution'].str.lower().isin(['yes', 'no'])].copy()

# Convert created_date to period representation for grouping (note: timezone info is dropped)
df_final.loc[:, 'year_month'] = df_final['created_date'].dt.to_period('M')

# Group by year-month and count questions
hist_data = df_final['year_month'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
hist_data.plot(kind='bar')
plt.xlabel('Year-Month')
plt.ylabel('Number of Questions')
plt.title('Histogram of Filtered Binary Questions by Creation Month')
plt.tight_layout()
plt.savefig("metaculus_binary_questions_histogram.png")

# Output the total number of filtered questions.
print("Total number of filtered questions:", len(df_final))


def old_format_forecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    zero_shot: bool = False
) -> str:
    """
    Format the prompt given the row data.
    """
    if zero_shot:
        return f"""You will be asked a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. 

Question: {question}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
"""
    else:
        # If not zero_shot, you can modify the prompt as needed.
        return f"""
Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}
"""


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



# Create dataset for huggingface
import pandas as pd
import re

# Randomly sample 1000 rows from df_final
# df_final = df_final.sample(n=1000)

data_list = []

cnt_test_questions = 0

for idx, row in df_final.iterrows():
    # Use title as the question, and body as the background.
    question = row["title"]
    background = row["body"]
    
    if background == "":
        background = "Not available"
    
    # Get resolution criteria and append fine_print if it exists
    metadata = row["metadata"]
    resolution_criteria = metadata["resolution_criteria"]
    fine_print = metadata["fine_print"]
    if fine_print:
        resolution_criteria += f" {fine_print}"
    
    # Skip practice questions
    if "practice" in question.lower():
        continue
    
    if  metadata["nr_forecasters"] < 3 :
        continue 
        
    # For date_begin and date_close, use open_time and scheduled_close_time respectively
    date_begin = (metadata["open_time"] if "open_time" in metadata else row["created_date"]).split("T")[0]
    date_close = metadata["scheduled_close_time"].split("T")[0]
    
    # if date is between 2023-07-01 and 2024-06-30, then it is a test question.
    # if date_begin >= "2023-06-01" and date_begin < "2024-06-30":
    #     cnt_test_questions += 1
    #     continue
    
    # if date_begin >= "2024-06-30":
    #     continue
    
    
    # if date_begin < "2025-05-01":
    #     continue
    
    # if date_close >= "2024-06-30":
    #     continue
    
    date_resolve = metadata["actual_resolve_time"].split("T")[0] if "actual_resolve_time" in metadata else None
    if date_resolve is None:
        date_resolve = metadata["scheduled_resolve_time"].split("T")[0] if "scheduled_resolve_time" in metadata else None
        
    # Extract URLs from background if any exist
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row["body"])
    
    # Convert resolution to binary (1 for yes, 0 for no)
    resolution = 1 if row["resolution"].lower() == "yes" else 0
    answer_type = "binary (yes/no)"
    answer = "YES" if row["resolution"].lower() == "yes" else "NO"
    
    # Check if question is resolved
    is_resolved = True if date_resolve else False
    
    # Create dictionary for this example
    example_dict = {
        'date_resolve_at': date_resolve,
        'date_begin': date_begin,
        'extracted_urls': urls,
        'question_type': row["question_type"],
        'url': row["url"],
        'background': background, # row["body"],
        'resolution_criteria': resolution_criteria,
        'is_resolved': is_resolved,
        'date_close': date_close,
        'question': row["title"],
        'data_source': row["data_source"],
        'resolution': resolution,
        'nr_forecasters': metadata["nr_forecasters"],
        'answer_type': answer_type,
        'answer': answer,
    }

    # Create the prompt (change zero_shot to True if desired)
    prompt = format_forecasting_prompt_binary(
        question_title=question,
        background=background,
        resolution_criteria=resolution_criteria,
    )
    # Retrieve the resolution field from the item.
    resolution = row["resolution"]
    
    # Append the prompt and resolution together.
    combined_output = prompt 
    example_dict['prompt'] = prompt
    
    data_list.append(example_dict)
    
print(len(data_list))

# Save data_list in .jsonl format
file_path = "/fast/nchandak/forecasting/datasets/metaculus/fromOct2025/binary_test.jsonl"
with open(file_path, 'w', encoding='utf-8') as f:
    print(f"Saving to {file_path} with data_list length {len(data_list)}")
    for item in data_list:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        
        
        
# Create Dataset
# ds = Dataset.from_dict(data_dict)

# Push train.json to hub
# ds.push_to_hub("nikhilchandak/metaculus-binary", "train")

# Pretty print first few rows of data_dict
# print("\nFirst few rows of data_dict:")
# for key in data_dict:
#     print(f"\n{key}:")
#     print(data_dict[key][:3])