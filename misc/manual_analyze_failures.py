#!/usr/bin/env python3
import json
import random
from collections import Counter

def load_jsonl(filepath: str):
    """Load JSONL file and return list of records."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def is_correct(record):
    """Check if any generation was correct according to score_Llama_4_Scout."""
    scores = record.get('score_Llama_4_Scout', [])
    if not scores:
        return False
    for score_dict in scores:
        if isinstance(score_dict, dict):
            if any(v == 1.0 for v in score_dict.values()):
                return True
    return False

def get_model_answer(record):
    """Extract the most common model answer."""
    extracted_answers = record.get('extracted_answer', [])
    all_extracted = []
    for ext_dict in extracted_answers:
        if isinstance(ext_dict, dict):
            all_extracted.extend(ext_dict.keys())
    if all_extracted:
        return Counter(all_extracted).most_common(1)[0][0]
    return "No answer"

def main():
    filepath = '/fast/nchandak/forecasting/evals/freeform/manual/news5-retrieval_1000/Qwen3-8B-sft-rl_eval_size_1000_generations_3_num_articles_5.jsonl'
    
    print("Loading data...")
    records = load_jsonl(filepath)
    print(f"Loaded {len(records)} records")
    
    # Get all failures
    failures = [r for r in records if not is_correct(r)]
    print(f"Found {len(failures)} failures")
    
    # Sample diverse failures for manual analysis
    # Get failures from different answer types
    failures_by_type = {}
    for failure in failures:
        answer_type = failure.get('answer_type', 'unknown')
        if answer_type not in failures_by_type:
            failures_by_type[answer_type] = []
        failures_by_type[answer_type].append(failure)
    
    # Sample from each type
    samples = []
    for answer_type, type_failures in failures_by_type.items():
        # Sample up to 3 from each type
        samples.extend(random.sample(type_failures, min(3, len(type_failures))))
    
    # Also add some random samples
    samples.extend(random.sample(failures, min(50, len(failures))))
    
    # Remove duplicates
    seen = set()
    unique_samples = []
    for sample in samples:
        idx = sample.get('idx', sample.get('question_id', None))
        if idx not in seen:
            seen.add(idx)
            unique_samples.append(sample)
    
    print(f"\nSampled {len(unique_samples)} unique failures for manual analysis\n")
    print("="*80)
    
    # Print samples for manual review
    for i, failure in enumerate(unique_samples[:100], 1):  # Limit to 100 for review
        print(f"\n{'='*80}")
        print(f"FAILURE #{i}")
        print(f"{'='*80}")
        print(f"Question: {failure.get('question_title', 'Unknown')}")
        print(f"Correct Answer: {failure.get('answer', 'Unknown')}")
        print(f"Model Answer: {get_model_answer(failure)}")
        print(f"Answer Type: {failure.get('answer_type', 'unknown')}")
        print(f"\nModel Response (first generation):")
        print("-" * 80)
        responses = failure.get('response', [])
        if responses:
            print(responses[0][:1000])  # First 1000 chars
        print("-" * 80)
        print()

if __name__ == '__main__':
    main()

