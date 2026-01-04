#!/usr/bin/env python3
"""
Manifold Markets Test Set Creator

Purpose:
    Creates clean test dataset from Manifold Markets binary questions with extensive filtering.
    Applies quality filters for volume, forecasters, dates, and content quality.

Filtering Steps:
    1. Filter to binary questions with yes/no resolutions
    2. Remove personal questions ("will my", "in my", "will i")
    3. Remove questions with links in background
    4. Filter by volume (>4000) and forecasters (>10)
    5. Filter by date (created after 2024-07-01)
    6. Remove platform-specific references (metaculus, manifold, polymarket)
    7. Ensure title length >= 15 chars and >= 5 words
    8. Remove empty questions/backgrounds

Output:
    - Binary test set in JSON format with prompts and metadata
    - Includes volume and forecaster statistics

Usage:
    python create_test_set.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Load the JSON data
with open("/fast/sgoel/forecasting/platforms/manifold_resolved.json", "r") as f:
    data = json.load(f)

# Load the JSON data
with open(f"/fast/nchandak/forecasting/datasets/manifold_resolved.json", "r") as f:
    data2 = json.load(f)

# Add the data2 to data
data.extend(data2)

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Keep only unique rows
# Count number of rows with duplicate id
# print(df[df.duplicated(subset=['id'])].shape[0])

# Keep only unique rows
df = df.drop_duplicates(subset=['id'])

# Print all the columns names 
print(df.columns)

# Convert 'created_date' to datetime (timezone-aware)
df['created_date'] = pd.to_datetime(df['created_date'], format='ISO8601')

# Create timezone-aware datetime objects for filtering
start_date = pd.Timestamp("2024-07-01", tz='UTC')
end_date = pd.Timestamp("2024-12-31 23:59:59", tz='UTC')

# No end date
end_date = pd.Timestamp("2025-12-31 23:59:59", tz='UTC')

# Print df length
print("OG length:", len(df))

# Print the different question types and their counts
print(df['question_type'].value_counts())

# Filter for binary questions and make a copy
df_binary = df[df['question_type'] == "binary"].copy()

# Print df_binary length
print("Binary length:", len(df_binary))

# Filter for questions created within the specified date range and make a copy
# df_date_filtered = df_binary[(df_binary['created_date'] >= start_date) & 
#                              (df_binary['created_date'] <= end_date)].copy()

# # Use .loc for assignment: Extract 'nr_forecasters' from the metadata
# df_date_filtered.loc[:, 'nr_forecasters'] = df_date_filtered['metadata'].apply(lambda x: x.get('nr_forecasters', 0))

# # Filter for questions with more than 10 forecasters.
# df_final = df_date_filtered[df_date_filtered['nr_forecasters'] > 10].copy()

# df_final = df_date_filtered.copy()
df_final = df_binary.copy()

# Print resolution value counts before filtering
print("\nResolution value counts before filtering:")
print(df_final['resolution'].value_counts())

# Filter for questions with resolution as yes or no (case-insensitive)
df_final = df_final[df_final['resolution'].str.lower().isin(['yes', 'no'])].copy()

# Convert created_date to year-month format while preserving timezone info
df_final.loc[:, 'year_month'] = df_final['created_date'].dt.strftime('%Y-%m')

# Create cumulative plots for both volume and number of forecasters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot for volume
volumes = df_final['metadata'].apply(lambda x: x.get('volume', 0))
# volume_bins = np.logspace(np.log10(1), np.log10(volumes.max()), 20)  # 20 log-spaced bins starting from $1
# volume_counts = [np.sum(volumes <= edge) for edge in volume_bins]

# ax1.bar(volume_bins[:-1], volume_counts[:-1], width=np.diff(volume_bins), align='edge')
# ax1.set_xscale('log')
# ax1.set_xlabel('Volume ($)')
# ax1.set_ylabel('Number of Questions with Volume ≤ X')
# ax1.set_title('Cumulative Distribution of Questions by Volume')
# ax1.grid(True, which="both", ls="-", alpha=0.2)

# # Plot for number of forecasters
# forecasters = df_final['metadata'].apply(lambda x: x.get('nr_forecasters', 0))
# forecaster_bins = np.logspace(np.log10(1), np.log10(forecasters.max()), 20)  # 20 log-spaced bins starting from 1
# forecaster_counts = [np.sum(forecasters <= edge) for edge in forecaster_bins]

# ax2.bar(forecaster_bins[:-1], forecaster_counts[:-1], width=np.diff(forecaster_bins), align='edge')
# ax2.set_xscale('log')
# ax2.set_xlabel('Number of Forecasters')
# ax2.set_ylabel('Number of Questions with Forecasters ≤ X')
# ax2.set_title('Cumulative Distribution of Questions by Forecasters')
# ax2.grid(True, which="both", ls="-", alpha=0.2)

# plt.tight_layout()
# plt.savefig("manifold_binary_questions_distributions.png")

# Only keep rows with date begin > 2023 
# df_final = df_final[df_final['created_date'] > pd.Timestamp("2023-01-01", tz='UTC')].copy()



# only keep questions with volume > 5 * 10^4 
# df_final = df_final[volumes > 5 * 10**3].copy()

# # Filter questions whose length is < 70 characters
# df_final = df_final[df_final['title'].str.len() > 50].copy()

# Print some questions with nr_forecasters < 2
# print("\nQuestions with less than 2 forecasters:")
# questions = df_final[df_final['metadata'].apply(lambda x: x.get('nr_forecasters', 0)) < 2]['title'].head(10)
# for i, question in enumerate(questions, 1):
#     print(f"\n{i}. {question}")

# Print df_final length
print("Before nr_forecasters > 10:", len(df_final))

# Keep only questions with nr_forecasters > 10
df_final = df_final[df_final['metadata'].apply(lambda x: x.get('nr_forecasters', 0)) > 10].copy()

# Print df_final length
print("Filtered length after nr_forecasters > 10:", len(df_final))

# Remove rows whose questions or background that contain the text "will my", "in my" or "will i" (case insensitive)
df_final = df_final[~df_final['title'].str.lower().str.contains("will my")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("will my")].copy()
df_final = df_final[~df_final['title'].str.lower().str.contains("in my")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("in my")].copy()
df_final = df_final[~df_final['title'].str.lower().str.contains("of my")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("of my")].copy()
df_final = df_final[~df_final['title'].str.lower().str.contains("will i")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("will i")].copy()
df_final = df_final[~df_final['title'].str.lower().str.contains("i will")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("i will")].copy()
df_final = df_final[~df_final['title'].str.lower().str.contains("i shall")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("i shall")].copy()

# Print df_final length
print("Filtered length after removing Will my, In my, Will I, I will, I shall:", len(df_final))


# Ensure that are at least 5 words in the title
df_final = df_final[df_final['title'].str.split().str.len() >= 5].copy()

# Print df_final length
print("Filtered length after ensuring at least 5 words in title:", len(df_final))

# Remove rows whose question or background contain ambiguous unicode characters
df_final = df_final[~df_final['title'].str.contains("[\u200B-\u200D\uFEFF]")].copy()
df_final = df_final[~df_final['body'].str.contains("[\u200B-\u200D\uFEFF]")].copy()


# Print df_final length
print("Filtered length after removing ambiguous unicode characters:", len(df_final))


# Remove rows whose background contain links 
df_final = df_final[~df_final['body'].str.contains("http")].copy()

# Print df_final length
print("Filtered length after removing links in background:", len(df_final))

# Remove rows whose questions or background are empty ("")
df_final = df_final[df_final['title'] != ""].copy()
df_final = df_final[df_final['body'] != ""].copy()

# Print df_final length
print("Filtered length after removing empty questions and background:", len(df_final))

# df_final = df_final[volumes > 100].copy()
# print("Filtered length after volumes > 100:", len(df_final))


df_final = df_final[df_final['title'].str.len() >= 15].copy()
print("Filtered length after title length >= 15:", len(df_final))


# Only keep rows with volume > 4000
df_final = df_final[df_final['metadata'].apply(lambda x: x.get('volume', 0)) > 4000].copy()
print("Filtered length after volume > 4000:", len(df_final))

# Only keep rows with created_date > 2024-07-01
df_final = df_final[df_final['created_date'] > pd.Timestamp("2024-07-01", tz='UTC')].copy()
print("Filtered length after created_date > 2024-07-01:", len(df_final))

# Remove questions which contain 'metaculus' in either the title or body
df_final = df_final[~df_final['title'].str.lower().str.contains("metaculus")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("metaculus")].copy()

# Print df_final length
print("Filtered length after removing metaculus:", len(df_final))


df_final = df_final[~df_final['title'].str.lower().str.contains("manifold")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("manifold")].copy()

# Print df_final length
print("Filtered length after removing manifold:", len(df_final))

df_final = df_final[~df_final['title'].str.lower().str.contains("polymarket")].copy()
df_final = df_final[~df_final['body'].str.lower().str.contains("polymarket")].copy()

# Print df_final length
print("Filtered length after removing polymarket:", len(df_final))


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
Question Background: {background}
Question close date: {date_close}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
"""
    else:
        # If not zero_shot, you can modify the prompt as needed.
        return f"""
Question: {question}
Background: {background}
Question close date: {date_close}
"""

# Create dataset for huggingface
import pandas as pd

# Randomly sample 1000 rows from df_final
# df_final = df_final.sample(n=1000)

data_list = []

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
        
    # For date_begin and date_close, use open_time and scheduled_close_time respectively
    date_begin = (metadata["open_time"] if "open_time" in metadata else row["created_date"]).split("T")[0]
    date_close = metadata["scheduled_close_time"].split("T")[0]
    
    date_resolve = metadata["actual_resolve_time"].split("T")[0] if "actual_resolve_time" in metadata else None
    if date_resolve is None:
        date_resolve = metadata["scheduled_resolve_time"].split("T")[0] if "scheduled_resolve_time" in metadata else None
        
    # Extract URLs from background if any exist
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row["body"])
    
    # Convert resolution to binary (1 for yes, 0 for no)
    resolution = 1 if row["resolution"].lower() == "yes" else 0
    
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
        'volume': metadata.get("volume", None),
        'id': row["id"],
        'post_id': metadata.get("post_id", None),
    }
    
    # assert id same as post_id
    assert example_dict['id'] == example_dict['post_id']

    # Create the prompt (change zero_shot to True if desired)
    prompt = format_forecasting_prompt(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=str(date_begin),
        date_close=str(date_close),
        zero_shot=False,
    )
    # Retrieve the resolution field from the item.
    resolution = row["resolution"]
    
    # Append the prompt and resolution together.
    combined_output = prompt 
    example_dict['prompt'] = prompt
    
    full_prompt = format_forecasting_prompt(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=str(date_begin),
        date_close=str(date_close),
        zero_shot=True,
    )
    example_dict['full_prompt'] = full_prompt
    data_list.append(example_dict)
    
# Only keep a random 1000 rows
import random
random.shuffle(data_list)
random_sample = data_list[:10]
for example in random_sample:
    # print(example['prompt'])
    print("Date resolve at:", example['date_resolve_at'])
    print(example['prompt'])
    print(f"Resolution Criteria: {example['resolution_criteria']}")
    print("Data source:", example['data_source'])
    print("\nResolution:", example['resolution'])
    # Volume and nr_forecasters
    print("\nVolume:", example['volume'])
    print("\nNumber of forecasters:", example['nr_forecasters'])
    # print(example['idx'])
    print("-"*100)



# Save data_list in proper format
file_path = "/fast/nchandak/forecasting/datasets/manifold/binary_test.json"
with open(file_path, 'w') as f:
    print(f"Saving to {file_path} with data_list length {len(data_list)}")
    json.dump(data_list, f, indent=4, ensure_ascii=False)