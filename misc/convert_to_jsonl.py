import json
from datetime import datetime

# Load filtered questions
with open('data/metaculus/filtered_binary_questions.json', 'r') as f:
    questions = json.load(f)

print(f"Converting {len(questions)} questions to JSONL format...")

# Create output directory if it doesn't exist
import os
output_dir = '/fast/nchandak/forecasting/datasets/metaculus/fromOct2025'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'metaculus_test.jsonl')

# Convert to JSONL format
with open(output_file, 'w') as f:
    for q in questions:
        # Extract dates
        resolution_date = q.get('resolution_date', '')
        date_resolve_at = resolution_date[:10] if resolution_date else ''
        
        # Get open time or published time for date_begin
        open_time = q.get('metadata', {}).get('open_time', '')
        published_at = q.get('metadata', {}).get('published_at', '')
        created_date = q.get('created_date', '')
        date_begin_raw = open_time or published_at or created_date
        date_begin = date_begin_raw[:10] if date_begin_raw else ''
        
        # Get close time
        close_time = q.get('metadata', {}).get('actual_close_time') or q.get('metadata', {}).get('scheduled_close_time', '')
        date_close = close_time[:10] if close_time else ''
        
        # Extract URLs from body (simple extraction)
        body = q.get('body', '') or ''
        extracted_urls = []  # Could parse URLs from body if needed
        
        # Convert resolution yes/no to 1/0
        resolution_str = q.get('resolution', '').lower()
        resolution_int = 1 if resolution_str == 'yes' else 0
        answer = 'YES' if resolution_str == 'yes' else 'NO'
        
        # Get metadata
        nr_forecasters = q.get('metadata', {}).get('nr_forecasters', 0) or 0
        resolution_criteria = q.get('metadata', {}).get('resolution_criteria', '')
        fine_print = q.get('metadata', {}).get('fine_print', '')
        
        # Create the prompt (similar to the reference format)
        prompt = f"""You will be asked a binary forecasting question.  You have to come up with the best estimate for whether the event asked in the question happens or happened. Please provide your reasoning before stating how likely is the event asked in the question to happen (your confidence of it resolving YES).
        
Question Title: {q.get('title', '')}
Question Background: {body}
Resolution Criteria: {resolution_criteria}

Think step by step about the information provided, reason about uncertainty and put your final confidence for the event asked in the question to resolve YES in <probability> </probability> tags. The probability should be a number between 0 and 1.

You will be rewarded based on the probability (p) you assign to your answer. Your answer will be evaluated using the BRIER SCORING RULE which is basically - (1 - p)^2 if your answer is correct and (- (p^2)) if your answer is incorrect. For example, if p = 0.6, and the event resolves to NO, then your score will be (- (0.6^2)) = -0.36 whereas if the event resolves to YES, then your score would be - (1 - 0.6)^2 = -0.16. Thus, the range of the score is [-1, 0]. If you output probability more than 0.5, then it is assumed that you think the event will likely resolve to "YES" while if you output probability less than 0.5, then it is assumed that you think the event will likely resolve to "NO". YOU HAVE TO MAXIMIZE YOUR BRIER SCORE.

Your final answer should be the probability that the event asked will resolve to YES and your response SHOULD STRICTLY END with <probability> </probability> tags.
"""
        
        # Create the JSONL entry
        jsonl_entry = {
            "date_resolve_at": date_resolve_at,
            "date_begin": date_begin,
            "extracted_urls": extracted_urls,
            "question_type": "binary",
            "url": q.get('url', ''),
            "background": body,
            "resolution_criteria": resolution_criteria,
            "is_resolved": True,
            "date_close": date_close,
            "question": q.get('title', ''),
            "data_source": "metaculus",
            "resolution": resolution_int,
            "nr_forecasters": nr_forecasters,
            "answer_type": "binary (yes/no)",
            "answer": answer,
            "prompt": prompt
        }
        
        # Write as single line JSON
        f.write(json.dumps(jsonl_entry) + '\n')

print(f"✓ Converted {len(questions)} questions")
print(f"✓ Saved to: {output_file}")

# Show sample
print(f"\n=== First question (formatted) ===")
with open(output_file, 'r') as f:
    first = json.loads(f.readline())
    print(f"Question: {first['question']}")
    print(f"Date begin: {first['date_begin']}")
    print(f"Date resolved: {first['date_resolve_at']}")
    print(f"Resolution: {first['answer']}")
    print(f"Forecasters: {first['nr_forecasters']}")
