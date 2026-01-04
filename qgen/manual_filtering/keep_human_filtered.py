"""
Filter questions to keep only those with human_filter == 1, validate resolution dates,
and ensure question_start_date is always earlier than resolution_date.
"""

import json
from datetime import datetime, timedelta


def load_jsonl(file_path: str):
    """Load JSONL file and return list of dictionaries."""
    data = []
    print(f"Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    print(f"  Loaded {len(data)} entries")
    return data


def save_jsonl(data, file_path: str):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} entries to {file_path}")


def parse_date(date_str: str):
    """Parse YYYY-MM-DD date string to datetime object."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def format_date(date_obj):
    """Format datetime object to YYYY-MM-DD string."""
    return date_obj.strftime('%Y-%m-%d')


def main():
    input_file = '/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-resolutionUpdated_humanFiltered.jsonl'
    output_file = '/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_test5news.jsonl'
    
    # Load data
    print("=" * 80)
    print("Loading input file")
    print("=" * 80)
    data = load_jsonl(input_file)
    
    # Filter by human_filter == 1
    print("\n" + "=" * 80)
    print("Filtering by human_filter == 1")
    print("=" * 80)
    before_filter = len(data)
    filtered_data = [entry for entry in data if entry.get('human_filter') == 1]
    print(f"  Before filter: {before_filter}")
    print(f"  After filter: {len(filtered_data)}")
    print(f"  Removed: {before_filter - len(filtered_data)}")
    
    # Validate resolution_date >= 2025-05-01
    print("\n" + "=" * 80)
    print("Validating resolution_date >= 2025-05-01")
    print("=" * 80)
    cutoff_date = parse_date('2025-05-01')
    invalid_entries = []
    
    for entry in filtered_data:
        resolution_date_str = entry.get('resolution_date')
        if not resolution_date_str:
            invalid_entries.append(entry)
            continue
        
        resolution_date = parse_date(resolution_date_str)
        if resolution_date is None:
            invalid_entries.append(entry)
            continue
        
        if resolution_date < cutoff_date:
            invalid_entries.append(entry)
    
    if invalid_entries:
        print(f"  ERROR: Found {len(invalid_entries)} entries with resolution_date < 2025-05-01 or invalid date")
        for entry in invalid_entries[:5]:  # Show first 5
            print(f"    qid: {entry.get('qid')}, resolution_date: {entry.get('resolution_date')}")
        assert False, f"Assertion failed: All resolution_date must be >= 2025-05-01"
    else:
        print(f"  ✓ All {len(filtered_data)} entries have resolution_date >= 2025-05-01")
    
    # Adjust question_start_date to be earlier than resolution_date
    print("\n" + "=" * 80)
    print("Adjusting question_start_date to be earlier than resolution_date")
    print("=" * 80)
    adjusted_count = 0
    
    for entry in filtered_data:
        resolution_date_str = entry.get('resolution_date')
        question_start_date_str = entry.get('question_start_date')
        
        if not resolution_date_str or not question_start_date_str:
            continue
        
        resolution_date = parse_date(resolution_date_str)
        question_start_date = parse_date(question_start_date_str)
        
        if resolution_date is None or question_start_date is None:
            continue
        
        # Calculate the maximum allowed question_start_date (resolution_date - 1 day)
        max_allowed_date = resolution_date - timedelta(days=1)
        
        # Update question_start_date to be min(question_start_date, resolution_date - 1)
        if question_start_date > max_allowed_date:
            entry['question_start_date'] = format_date(max_allowed_date)
            adjusted_count += 1
    
    print(f"  Adjusted {adjusted_count} entries where question_start_date was >= resolution_date")
    
    # Remove human_filter attribute since all entries pass the filter
    print("\n" + "=" * 80)
    print("Removing human_filter attribute")
    print("=" * 80)
    for entry in filtered_data:
        if 'human_filter' in entry:
            del entry['human_filter']
    print(f"  Removed human_filter from all {len(filtered_data)} entries")
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)
    save_jsonl(filtered_data, output_file)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nOutput file: {output_file}")
    print(f"Total questions: {len(filtered_data)}")


if __name__ == "__main__":
    main()

