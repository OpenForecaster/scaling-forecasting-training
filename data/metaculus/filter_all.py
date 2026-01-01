#!/usr/bin/env python3
"""
Metaculus Question Filter and Cleaner

Purpose:
    Filters and cleans Metaculus questions by removing duplicates, market-specific questions,
    and questions with specific formatting issues.

Filtering Steps:
    1. Remove duplicate questions (by question title)
    2. Remove market close price questions
    3. Remove questions with options in parentheses at end
    4. Remove very short questions (<20 chars)
    5. Remove questions with specific keywords

Input:
    - imp_questions_filtered.jsonl: Pre-filtered Metaculus questions

Output:
    - imp_questions_final_clean.jsonl: Cleaned and deduplicated questions

Usage:
    python filter_all.py
"""

import json
import re

# Configuration
input_file = '/home/nchandak/forecasting/data/metaculus/imp_questions_filtered.jsonl'
output_file = '/home/nchandak/forecasting/data/metaculus/imp_questions_final_clean.jsonl'

# Load the data
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Initial dataset size: {len(data)}")

# Step 1: Remove duplicates (keep first occurrence)
print("\n=== Step 1: Removing Duplicates ===")
seen_questions = set()
data_no_duplicates = []
duplicates_removed = 0

for item in data:
    question = item.get('question_title', item.get('question'))
    if question not in seen_questions:
        seen_questions.add(question)
        data_no_duplicates.append(item)
    else:
        duplicates_removed += 1

print(f"Duplicates removed: {duplicates_removed}")
print(f"After removing duplicates: {len(data_no_duplicates)}")

# Step 2: Remove market close price questions
print("\n=== Step 2: Removing Market Close Price Questions ===")
market_keywords = ['close price', 'closing price', 'close at', 'close above', 'close below', 'market close']
data_no_market = []
market_removed = 0

for item in data_no_duplicates:
    question = item.get('question_title', item.get('question'))
    question_lower = question.lower()
    if not any(kw in question_lower for kw in market_keywords):
        data_no_market.append(item)
    else:
        market_removed += 1

print(f"Market close price questions removed: {market_removed}")
print(f"After removing market questions: {len(data_no_market)}")

# Step 3: Remove questions with options in parentheses at the end
print("\n=== Step 3: Removing Questions with Options in Parentheses ===")
# Pattern to match questions with options in parentheses at the end
pattern = r'\([^)]+\)[?!]?\s*$'
data_no_options = []
options_removed = 0

for item in data_no_market:
    question = item.get('question_title', item.get('question'))
    if not re.search(pattern, question):
        data_no_options.append(item)
    else:
        options_removed += 1

print(f"Questions with options in parentheses removed: {options_removed}")
print(f"After removing option questions: {len(data_no_options)}")

# Step 4: Remove meta prediction questions
print("\n=== Step 4: Removing Meta Prediction Questions ===")
meta_keywords = ['community prediction', 'metaculus question', 'community median', 'community forecast']
data_no_meta = []
meta_removed = 0

for item in data_no_options:
    question = item.get('question_title', item.get('question'))
    question_lower = question.lower()
    if not any(kw in question_lower for kw in meta_keywords):
        data_no_meta.append(item)
    else:
        meta_removed += 1

print(f"Meta prediction questions removed: {meta_removed}")
print(f"After removing meta questions: {len(data_no_meta)}")

# Step 5: Remove share/stock price questions
print("\n=== Step 5: Removing Share/Stock Price Questions ===")
price_keywords = ['share price', 'share at', 'stock price', 'stock at', 'hit $', 'reach $', 'trading at']
data_clean = []
price_removed = 0

for item in data_no_meta:
    question = item.get('question_title', item.get('question'))
    question_lower = question.lower()
    if not any(kw in question_lower for kw in price_keywords):
        data_clean.append(item)
    else:
        price_removed += 1

print(f"Share/stock price questions removed: {price_removed}")
print(f"Final dataset size: {len(data_clean)}")

# Print final statistics
yes_count = sum(1 for item in data_clean if item['answer'] == 'YES')
no_count = sum(1 for item in data_clean if item['answer'] == 'NO')
avg_forecasters = sum(item['nr_forecasters'] for item in data_clean) / len(data_clean) if data_clean else 0

print(f"\n{'='*50}")
print(f"=== Final Dataset Statistics ===")
print(f"{'='*50}")
print(f"Total questions: {len(data_clean)}")
print(f"Questions removed: {len(data) - len(data_clean)} ({(len(data) - len(data_clean))/len(data)*100:.1f}%)")
print(f"YES resolutions: {yes_count} ({yes_count/len(data_clean)*100:.1f}%)")
print(f"NO resolutions: {no_count} ({no_count/len(data_clean)*100:.1f}%)")
print(f"Average forecasters per question: {avg_forecasters:.2f}")

# Summary of removals
print(f"\n=== Removal Summary ===")
print(f"Duplicates: {duplicates_removed}")
print(f"Market close price: {market_removed}")
print(f"Options in parentheses: {options_removed}")
print(f"Meta predictions: {meta_removed}")
print(f"Share/stock prices: {price_removed}")
print(f"Total removed: {duplicates_removed + market_removed + options_removed + meta_removed + price_removed}")

# Save the cleaned data
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data_clean:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nFinal cleaned dataset saved to: {output_file}")

# Print first 5 examples
print(f"\n{'='*50}")
print(f"=== First 5 Examples ===")
print(f"{'='*50}")
for i, item in enumerate(data_clean[:5], 1):
    question = item.get('question_title', item.get('question'))
    print(f"\n{i}. Question: {question}")
    print(f"   Answer: {item['answer']}")
    print(f"   Forecasters: {item['nr_forecasters']}")
    print(f"   Date resolved: {item.get('date_resolve_at', 'N/A')}")

# Verification - ensure no filtered items remain
print(f"\n{'='*50}")
print(f"=== Verification ===")
print(f"{'='*50}")
meta_remaining = sum(1 for item in data_clean if any(kw in item.get('question_title', item.get('question')).lower() for kw in meta_keywords))
price_remaining = sum(1 for item in data_clean if any(kw in item.get('question_title', item.get('question')).lower() for kw in price_keywords))
market_remaining = sum(1 for item in data_clean if any(kw in item.get('question_title', item.get('question')).lower() for kw in market_keywords))
options_remaining = sum(1 for item in data_clean if re.search(pattern, item.get('question_title', item.get('question'))))

print(f"Meta prediction questions remaining: {meta_remaining}")
print(f"Share/stock price questions remaining: {price_remaining}")
print(f"Market close price questions remaining: {market_remaining}")
print(f"Questions with options remaining: {options_remaining}")

if meta_remaining == 0 and price_remaining == 0 and market_remaining == 0 and options_remaining == 0:
    print("\n✓ All filters applied successfully - no unwanted questions remain!")
else:
    print("\n⚠ Warning: Some unwanted questions may still remain")

