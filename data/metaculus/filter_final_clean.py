import json
import re

# Load the data
input_file = '/home/nchandak/forecasting/data/metaculus/imp_questions_filtered_clean.jsonl'
output_file = '/home/nchandak/forecasting/data/metaculus/imp_questions_final.jsonl'

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Initial dataset size: {len(data)}")

# Pattern to match questions with options in parentheses at the end
# Matches: (something) possibly followed by ? or ! at the end
pattern = r'\([^)]+\)[?!]?\s*$'

data_clean = []
questions_with_options_removed = 0

for item in data:
    question = item.get('question_title', item.get('question'))  # Handle both field names
    if not re.search(pattern, question):
        data_clean.append(item)
    else:
        questions_with_options_removed += 1

print(f"Questions with options in parentheses removed: {questions_with_options_removed}")
print(f"Final dataset size: {len(data_clean)}")

# Print statistics
yes_count = sum(1 for item in data_clean if item['answer'] == 'YES')
no_count = sum(1 for item in data_clean if item['answer'] == 'NO')
avg_forecasters = sum(item['nr_forecasters'] for item in data_clean) / len(data_clean) if data_clean else 0

print(f"\n=== Final Dataset Statistics ===")
print(f"Total questions: {len(data_clean)}")
print(f"YES resolutions: {yes_count} ({yes_count/len(data_clean)*100:.1f}%)")
print(f"NO resolutions: {no_count} ({no_count/len(data_clean)*100:.1f}%)")
print(f"Average forecasters per question: {avg_forecasters:.2f}")

# Save the cleaned data
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data_clean:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nFinal dataset saved to: {output_file}")

# Print first 5 examples
print("\n=== First 5 Examples ===")
for i, item in enumerate(data_clean[:5], 1):
    question = item.get('question_title', item.get('question'))
    print(f"\n{i}. Question: {question}")
    print(f"   Answer: {item['answer']}")
    print(f"   Forecasters: {item['nr_forecasters']}")

# Verify no questions with options remain
print("\n=== Verification ===")
remaining_with_options = [item for item in data_clean if re.search(pattern, item.get('question_title', item.get('question')))]
print(f"Questions with options remaining: {len(remaining_with_options)}")

