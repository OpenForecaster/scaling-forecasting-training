import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
# Load the JSON data
with open("metaculus_resolved_filtered.json", "r") as f:
    data = json.load(f)

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
df_binary = df[df['question_type'] == "multiple_choice"].copy()

# Filter for questions created within the specified date range and make a copy
# df_date_filtered = df_binary[(df_binary['created_date'] >= start_date) & 
#                              (df_binary['created_date'] <= end_date)].copy()

# # Use .loc for assignment: Extract 'nr_forecasters' from the metadata
# df_date_filtered.loc[:, 'nr_forecasters'] = df_date_filtered['metadata'].apply(lambda x: x.get('nr_forecasters', 0))

# # Filter for questions with more than 10 forecasters.
# df_final = df_date_filtered[df_date_filtered['nr_forecasters'] > 10].copy()

df_final = df_binary.copy()

# Filter for questions with resolution as not annulled
df_final = df_final[df_final['resolution'] != "annulled"].copy()

# Print resolution value counts before filtering
print("\nResolution value counts before filtering:")
print(df_final['resolution'].value_counts())

# Convert created_date to period representation for grouping (note: timezone info is dropped)
df_final.loc[:, 'year_month'] = df_final['created_date'].dt.to_period('M')

# Group by year-month and count questions
hist_data = df_final['year_month'].value_counts().sort_index()


# Output the total number of filtered questions.
print("Total number of filtered questions:", len(df_final))


# print column names
print(df_final.columns)

# for idx, row in df_final.iterrows():
#     for col in row.index:
#         print(col, ":", row[col])
#     print("-"*100)


def format_forecasting_prompt(
    question: str,
    background: str,
    resolution_criteria: str,
    date_begin: str,
    date_close: str,
    zero_shot: bool = False,
    options: list[str] = [],
) -> str:
    """
    Format the prompt given the row data.
    """
    
    middle_text = ""
    for i, option in enumerate(options):
        middle_text += f"{chr(i + ord('A'))}. {option}\n"
    
    if zero_shot:
        return f"""You will be asked a forecasting question. You have to come up with the best estimate for whether the event asked in the question happens or happened. 

Question: {question}
{middle_text}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}

Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. YOUR FINAL PREDICTION SHOULD STRICTLY BE BETWEEN 0 AND 1. For example, if you believe the answer is 75% likely, you would write *0.75*. MAKE SURE TO FORMAT IT CORRECTLY AND PLACE BETWEEN ASTERISKS.
"""
    else:
        # If not zero_shot, you can modify the prompt as needed.
        return f"""
Question: {question}
{middle_text}
Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}
"""

# Create dataset for huggingface
import pandas as pd

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
        
    # For date_begin and date_close, use open_time and scheduled_close_time respectively
    date_begin = (metadata["open_time"] if "open_time" in metadata else row["created_date"]).split("T")[0]
    date_close = metadata["scheduled_close_time"].split("T")[0]
    
    # MCQ specific
    # options = metadata["options"]
    # answer = row["resolution"]
    # answer_idx = options.index(answer)
    
    # choice = chr(answer_idx + ord('A'))
    
    # MCQ specific
    options = metadata["options"]
    answer = row["resolution"]
    
    # If number of options is > 4, randomly keep 4 options (with the answer being one of the options)
    if len(options) > 4:
        options.pop(options.index(answer))
        options = random.sample(options, 3)
        options.append(answer)
        # Shuffle the options
        random.shuffle(options)
    
    answer_idx = options.index(answer)
    
    # assert answer_idx is not None 
    assert answer_idx is not None
    choice = chr(answer_idx + ord('A'))
    
    
    # if date is between 2023-07-01 and 2024-06-30, then it is a test question.
    # if date_begin >= "2023-06-01" and date_begin < "2024-06-30":
    #     cnt_test_questions += 1
    #     continue
    
    # if date_begin >= "2024-06-30":
    #     continue
    
    # if date_close >= "2024-06-30":
    #     continue
    
    date_resolve = metadata["actual_resolve_time"].split("T")[0] if "actual_resolve_time" in metadata else None
    if date_resolve is None:
        date_resolve = metadata["scheduled_resolve_time"].split("T")[0] if "scheduled_resolve_time" in metadata else None
        
    # Extract URLs from background if any exist
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row["body"])

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
        'resolution': choice,
        'nr_forecasters': metadata["nr_forecasters"],
        "options": options,
        "answer": answer,
        "answer_idx": answer_idx,
    }

    # Create the prompt (change zero_shot to True if desired)
    prompt = format_forecasting_prompt(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=str(date_begin),
        date_close=str(date_close),
        zero_shot=False,
        options=options,
    )
    # Retrieve the resolution field from the item.
    resolution = row["resolution"]
    
    # Append the prompt and resolution together.
    combined_output = prompt 
    example_dict['prompt'] = prompt
    
    data_list.append(example_dict)
    
print(f"Number of test questions: {cnt_test_questions}")
print(len(data_list))



# random_sample = data_list[:10]
# for example in random_sample:
#     # print(example['prompt'])
#     print("Date resolve at:", example['date_resolve_at'])
#     print(example['prompt'])
#     print("\nResolution:", example['resolution'])
#     # print(example['idx'])
#     print("-"*100)


# Save data_list in proper format
file_path = "/fast/nchandak/forecasting/datasets/metaculus/mcq_raw.json"
with open(file_path, 'w') as f:
    print(f"Saving to {file_path} with data_list length {len(data_list)}")
    json.dump(data_list, f, indent=4, ensure_ascii=False)
    
# Create Dataset
# ds = Dataset.from_dict(data_dict)

# Push train.json to hub
# ds.push_to_hub("nikhilchandak/metaculus-binary", "train")

# Pretty print first few rows of data_dict
# print("\nFirst few rows of data_dict:")
# for key in data_dict:
#     print(f"\n{key}:")
#     print(data_dict[key][:3])