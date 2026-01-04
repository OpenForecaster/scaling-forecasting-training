import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
# plt.savefig("metaculus_binary_questions_histogram.png")

# Output the total number of filtered questions.
print("Total number of filtered questions:", len(df_final))


def format_forecasting_prompt(
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
        return f"""Forecasting Question:
Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {date_close}
"""

# print first few rows of df_final (fully)
# print(df_final.head())
# # Print column names of df_final
# print("\nColumn names in df_final:")
# print(df_final.columns)
# Print a sample metadata entry
# if 'metadata' in df_final.columns:
#     print("\nSample metadata entry:")
#     sample_metadata = df_final['metadata'].iloc[0]
#     print(json.dumps(sample_metadata, indent=2))
# else:
#     print("\nNo metadata column found in df_final")

# Create dataset for huggingface
from datasets import Dataset
import pandas as pd

data_dict = {
    'date_resolve_at': [],
    'date_begin': [],
    'extracted_urls': [], 
    'question_type': [],
    'url': [],
    'background': [],
    'resolution_criteria': [],
    'is_resolved': [],
    'date_close': [],
    'question': [],
    'data_source': [],
    'resolution': [],
    'nr_forecasters': []
}

prompts = []
for _, row in df_final.iterrows():
    # Use title as the question, and body as the background.
    question = row["title"]
    background = row["body"]
    
    # Get resolution criteria and append fine_print if it exists
    metadata = row["metadata"]
    resolution_criteria = metadata["resolution_criteria"]
    fine_print = metadata["fine_print"]
    if fine_print:
        resolution_criteria += f" {fine_print}"
        
    # For date_begin and date_close, use open_time and scheduled_close_time respectively
    date_begin = (metadata["open_time"] if "open_time" in metadata else row["created_date"]).split("T")[0]
    date_close = metadata["scheduled_close_time"].split("T")[0]
    
    # Only keep questions created before June 30, 2024
    if date_begin >= "2024-06-30":
        continue
    
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
    
    # Add to dictionary
    data_dict['date_resolve_at'].append(date_resolve)
    data_dict['date_begin'].append(date_begin)
    data_dict['extracted_urls'].append(urls)
    data_dict['question_type'].append(row["question_type"])
    data_dict['url'].append(row["url"])
    data_dict['background'].append(row["body"])
    data_dict['resolution_criteria'].append(resolution_criteria) #metadata["resolution_criteria"])
    data_dict['is_resolved'].append(is_resolved)
    data_dict['date_close'].append(date_close)
    data_dict['question'].append(row["title"])
    data_dict['data_source'].append(row["data_source"])
    data_dict['resolution'].append(resolution)
    # Add number of forecasters
    data_dict['nr_forecasters'].append(metadata["nr_forecasters"])

    
    # Create the prompt (change zero_shot to True if desired)
    prompt = format_forecasting_prompt(
        question=question,
        background=background,
        resolution_criteria=resolution_criteria,
        date_begin=str(date_begin),
        date_close=str(date_close),
        zero_shot=True
    )
    # Retrieve the resolution field from the item.
    resolution = row["resolution"]
    
    # Append the prompt and resolution together.
    combined_output = prompt + "\nResolution: " + resolution
    prompts.append(combined_output)

# For demonstration, print out the first few prompts.
for p in prompts[:3]:
    print(p)

print(len(prompts))
# Save data_dict as train.json
# with open('train.json', 'w') as f:
#     json.dump(data_dict, f)

# Create Dataset
# ds = Dataset.from_dict(data_dict)

# Push train.json to hub
# ds.push_to_hub("nikhilchandak/metaculus-binary", "train")

# Pretty print first few rows of data_dict
# print("\nFirst few rows of data_dict:")
# for key in data_dict:
#     print(f"\n{key}:")
#     print(data_dict[key][:3])