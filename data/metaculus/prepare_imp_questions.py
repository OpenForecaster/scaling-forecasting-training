import json
import pandas as pd
from datetime import datetime
import re

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping histogram generation")

# Load data from .json file
json_path = "/home/nchandak/forecasting/data/metaculus/imp_questions_after_May2025.json"
print(f"Loading data from {json_path}")
with open(json_path, "r") as f:
    data = json.load(f)

print(f"Total questions loaded: {len(data)}")

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Filter for binary questions
df_binary = df[df['question_type'] == "binary"].copy()
print(f"Binary questions: {len(df_binary)}")

# Print resolution value counts before filtering
print("\nResolution value counts before filtering:")
print(df_binary['resolution'].value_counts(dropna=False))

# Filter for questions with resolution as yes or no (case-insensitive) - must be resolved
df_resolved = df_binary[df_binary['resolution'].notna()].copy()
df_resolved = df_resolved[df_resolved['resolution'].str.lower().isin(['yes', 'no'])].copy()
print(f"Binary questions with yes/no resolution: {len(df_resolved)}")

# Filter for questions resolved after May 01, 2025
# Extract actual_resolve_time from metadata
df_resolved['actual_resolve_time'] = df_resolved['metadata'].apply(
    lambda x: x.get('actual_resolve_time', None)
)

# Filter out questions without actual_resolve_time
df_with_resolve_time = df_resolved[df_resolved['actual_resolve_time'].notna()].copy()
print(f"Questions with actual_resolve_time: {len(df_with_resolve_time)}")

# Convert actual_resolve_time to datetime
df_with_resolve_time['resolve_datetime'] = pd.to_datetime(df_with_resolve_time['actual_resolve_time'])

# Filter for questions resolved after May 01, 2025
cutoff_date = pd.Timestamp("2025-05-01", tz='UTC')
df_final = df_with_resolve_time[df_with_resolve_time['resolve_datetime'] >= cutoff_date].copy()

print(f"\nFinal filtered questions (resolved after May 01, 2025): {len(df_final)}")

# Print resolution value counts after filtering
print("\nResolution value counts after filtering:")
print(df_final['resolution'].value_counts())

# Convert resolve_datetime to period representation for grouping
df_final.loc[:, 'year_month'] = df_final['resolve_datetime'].dt.to_period('M')

# Group by year-month and count questions
hist_data = df_final['year_month'].value_counts().sort_index()

if HAS_MATPLOTLIB:
    plt.figure(figsize=(10, 6))
    hist_data.plot(kind='bar')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Questions')
    plt.title('Histogram of Resolved Binary Questions by Resolution Month (After May 2025)')
    plt.tight_layout()
    plt.savefig("/home/nchandak/forecasting/data/metaculus/imp_questions_histogram.png")
    print("Histogram saved to imp_questions_histogram.png")
else:
    print("Skipping histogram generation (matplotlib not available)")

# Output the total number of filtered questions.
print(f"\nTotal number of filtered questions: {len(df_final)}")


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
data_list = []

for idx, row in df_final.iterrows():
    # Use title as the question, and body as the background.
    question = row["title"]
    background = row["body"]
    
    if background == "":
        background = "Not available"
    
    # Get resolution criteria and append fine_print if it exists
    metadata = row["metadata"]
    resolution_criteria = metadata.get("resolution_criteria", "")
    fine_print = metadata.get("fine_print", "")
    if fine_print:
        resolution_criteria += f" {fine_print}"
    
    # Skip practice questions
    if "practice" in question.lower():
        continue
    
    if metadata.get("nr_forecasters", 0) < 3:
        continue 
        
    # For date_begin and date_close, use open_time and scheduled_close_time respectively
    date_begin = (metadata.get("open_time", row["created_date"])).split("T")[0]
    date_close = metadata.get("scheduled_close_time", "").split("T")[0] if metadata.get("scheduled_close_time") else None
    
    date_resolve = metadata.get("actual_resolve_time", "").split("T")[0] if metadata.get("actual_resolve_time") else None
    if date_resolve is None:
        date_resolve = metadata.get("scheduled_resolve_time", "").split("T")[0] if metadata.get("scheduled_resolve_time") else None
        
    # Extract URLs from background if any exist
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', row["body"])
    
    # Convert resolution to binary (1 for yes, 0 for no)
    resolution = 1 if row["resolution"].lower() == "yes" else 0
    answer_type = "binary (yes/no)"
    answer = "YES" if row["resolution"].lower() == "yes" else "NO"
    
    # Check if question is resolved
    is_resolved = True
    
    # Create dictionary for this example
    example_dict = {
        'date_resolve_at': date_resolve,
        'date_begin': date_begin,
        'extracted_urls': urls,
        'question_type': row["question_type"],
        'url': row["url"],
        'background': background,
        'resolution_criteria': resolution_criteria,
        'is_resolved': is_resolved,
        'date_close': date_close,
        'question_title': question,
        'data_source': row["data_source"],
        'resolution': resolution,
        'nr_forecasters': metadata.get("nr_forecasters", 0),
        'answer_type': answer_type,
        'answer': answer,
    }

    # Create the prompt
    prompt = format_forecasting_prompt_binary(
        question_title=question,
        background=background,
        resolution_criteria=resolution_criteria,
    )
    
    example_dict['prompt'] = prompt
    
    data_list.append(example_dict)
    
print(f"\nFinal dataset size (after additional filtering): {len(data_list)}")

# Save data_list in .jsonl format
output_file_path = "/home/nchandak/forecasting/data/metaculus/imp_questions_filtered.jsonl"
with open(output_file_path, 'w', encoding='utf-8') as f:
    print(f"Saving to {output_file_path} with data_list length {len(data_list)}")
    for item in data_list:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nDataset saved successfully to {output_file_path}")

# Print some statistics
print("\n=== Dataset Statistics ===")
print(f"Total questions: {len(data_list)}")
print(f"YES resolutions: {sum(1 for item in data_list if item['answer'] == 'YES')}")
print(f"NO resolutions: {sum(1 for item in data_list if item['answer'] == 'NO')}")
print(f"Average forecasters per question: {sum(item['nr_forecasters'] for item in data_list) / len(data_list):.2f}")

# Print first example
if data_list:
    print("\n=== First Example ===")
    first_example = data_list[0]
    print(f"Question: {first_example['question_title']}")
    print(f"Answer: {first_example['answer']}")
    print(f"Date resolved: {first_example['date_resolve_at']}")
    print(f"Number of forecasters: {first_example['nr_forecasters']}")

