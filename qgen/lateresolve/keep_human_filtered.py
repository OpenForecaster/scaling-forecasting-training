"""
Filter and update questions based on human annotations and corrected resolution dates.

This script:
1. Loads file1 (evaluated questions with idx and correct resolution_date)
2. Loads file3 (original questions with all data, using qid)
3. Maps idx to qid using question_title matching
4. Takes the 540 questions from file3 that correspond to file1
5. Updates resolution_date from file1 (converts timestamp to YYYY-MM-DD)
6. Removes 'relevant_articles_sorted_by_docs' field
7. Checks file2 (human annotations) and keeps only questions with human_filter == 1 (if annotated)
8. Saves the filtered results
"""

import json
import argparse
from typing import List, Dict, Set
from datetime import datetime


def load_jsonl(file_path: str) -> List[Dict]:
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


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} entries to {file_path}")


def normalize_question(question: str) -> str:
    """Normalize question text for matching."""
    return question.strip().lower()


def create_idx_to_qid_mapping(file1_data: List[Dict], file3_data: List[Dict]) -> Dict[int, str]:
    """
    Create a mapping from idx (file1) to qid (file3) using question_title matching.
    
    Args:
        file1_data: List of questions from file1 with idx
        file3_data: List of questions from file3 with qid
        
    Returns:
        Dictionary mapping idx to qid
    """
    # Create a mapping from normalized question to qid
    question_to_qid = {}
    for entry in file3_data:
        qid = entry.get('qid')
        question_title = entry.get('question_title', '')
        if qid and question_title:
            normalized = normalize_question(question_title)
            question_to_qid[normalized] = str(qid)
    
    # Map idx to qid
    idx_to_qid = {}
    unmatched_count = 0
    
    for entry in file1_data:
        idx = entry.get('idx')
        question_title = entry.get('question_title', '')
        if idx is not None and question_title:
            normalized = normalize_question(question_title)
            if normalized in question_to_qid:
                idx_to_qid[idx] = question_to_qid[normalized]
            else:
                unmatched_count += 1
                if unmatched_count <= 5:  # Show first 5 unmatched
                    print(f"  Warning: No match found for idx {idx}: {question_title[:60]}...")
    
    print(f"\nMapping summary:")
    print(f"  Total in file1: {len(file1_data)}")
    print(f"  Matched: {len(idx_to_qid)}")
    print(f"  Unmatched: {unmatched_count}")
    
    return idx_to_qid


def get_resolution_dates_from_file1(file1_data: List[Dict]) -> Dict[int, int]:
    """
    Extract resolution dates from file1.
    
    Args:
        file1_data: List of questions from file1
        
    Returns:
        Dictionary mapping idx to resolution_date (timestamp)
    """
    idx_to_resolution_date = {}
    for entry in file1_data:
        idx = entry.get('idx')
        resolution_date = entry.get('resolution_date')
        if idx is not None and resolution_date is not None:
            idx_to_resolution_date[idx] = resolution_date
    return idx_to_resolution_date


def convert_timestamp_to_date(timestamp: int) -> str:
    """Convert Unix timestamp to YYYY-MM-DD format."""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    except (ValueError, OSError):
        return ""


def parse_date_to_timestamp(date_str: str) -> int:
    """Convert YYYY-MM-DD string to Unix timestamp."""
    try:
        return int(datetime.strptime(date_str, '%Y-%m-%d').timestamp())
    except (ValueError, TypeError):
        return 0


def get_human_filter_mapping(file2_data: List[Dict]) -> Dict[str, int]:
    """
    Get human_filter values from file2.
    
    Args:
        file2_data: List of questions from file2
        
    Returns:
        Dictionary mapping qid to human_filter value
    """
    qid_to_human_filter = {}
    for entry in file2_data:
        qid = str(entry.get('qid', ''))
        human_filter = entry.get('human_filter')
        if qid and human_filter is not None:
            qid_to_human_filter[qid] = human_filter
    return qid_to_human_filter


def filter_and_update_questions(
    file3_data: List[Dict],
    idx_to_qid: Dict[int, str],
    idx_to_resolution_date: Dict[int, int],
    qid_to_human_filter: Dict[str, int]
) -> List[Dict]:
    """
    Filter file3 data and update with corrected resolution dates.
    
    Args:
        file3_data: List of questions from file3
        idx_to_qid: Mapping from idx to qid
        idx_to_resolution_date: Mapping from idx to resolution_date (timestamp)
        qid_to_human_filter: Mapping from qid to human_filter value
        
    Returns:
        Filtered and updated list of questions
    """
    # Create reverse mapping: qid to idx
    qid_to_idx = {qid: idx for idx, qid in idx_to_qid.items()}
    
    # Get set of valid qids
    valid_qids = set(idx_to_qid.values())
    
    filtered = []
    stats = {
        'total_file3': 0,
        'in_valid_set': 0,
        'no_human_annotation': 0,
        'human_filter_pass': 0,
        'human_filter_fail': 0,
        'kept': 0,
    }
    
    for entry in file3_data:
        stats['total_file3'] += 1
        qid = str(entry.get('qid', ''))
        
        # Check if qid is in the valid set (from file1)
        if qid not in valid_qids:
            continue
        
        stats['in_valid_set'] += 1
        
        # Check human_filter from file2
        human_filter = qid_to_human_filter.get(qid)
        
        # Filter based on human annotation
        if human_filter is None:
            # Not annotated yet - keep it
            stats['no_human_annotation'] += 1
        elif human_filter == 1:
            # Passed human filtering - keep it
            stats['human_filter_pass'] += 1
        else:
            # Failed human filtering - skip it
            stats['human_filter_fail'] += 1
            continue
        
        # Make a copy to avoid modifying original
        updated_entry = entry.copy()
        
        # Remove relevant_articles_sorted_by_docs field
        if 'relevant_articles_sorted_by_docs' in updated_entry:
            del updated_entry['relevant_articles_sorted_by_docs']
        
        # Update resolution_date: take minimum of file1 and file3
        idx = qid_to_idx.get(qid)
        file1_timestamp = None
        file3_timestamp = None
        
        if idx is not None and idx in idx_to_resolution_date:
            file1_timestamp = idx_to_resolution_date[idx]
        
        # Get file3 timestamp (could be timestamp or string)
        file3_resolution_date = updated_entry.get('resolution_date')
        if isinstance(file3_resolution_date, int):
            file3_timestamp = file3_resolution_date
        elif isinstance(file3_resolution_date, str):
            file3_timestamp = parse_date_to_timestamp(file3_resolution_date)
        
        # Take minimum of the two timestamps
        if file1_timestamp and file3_timestamp:
            min_timestamp = min(file1_timestamp, file3_timestamp)
        elif file1_timestamp:
            min_timestamp = file1_timestamp
        elif file3_timestamp:
            min_timestamp = file3_timestamp
        else:
            min_timestamp = None
        
        # Convert to date string
        if min_timestamp:
            date_str = convert_timestamp_to_date(min_timestamp)
            if date_str:
                updated_entry['resolution_date'] = date_str
        
        # Convert question_start_date from timestamp to YYYY-MM-DD if it's a timestamp
        if 'question_start_date' in updated_entry:
            qsd = updated_entry['question_start_date']
            if isinstance(qsd, int):
                date_str = convert_timestamp_to_date(qsd)
                if date_str:
                    updated_entry['question_start_date'] = date_str
        
        # Add human_filter field if annotated
        if human_filter is not None:
            updated_entry['human_filter'] = human_filter
        
        filtered.append(updated_entry)
        stats['kept'] += 1
    
    print(f"\nFiltering and update statistics:")
    print(f"  Total entries in file3: {stats['total_file3']}")
    print(f"  In valid set (from file1): {stats['in_valid_set']}")
    print(f"  - Not annotated yet: {stats['no_human_annotation']}")
    print(f"  - Passed human filter (1): {stats['human_filter_pass']}")
    print(f"  - Failed human filter (0): {stats['human_filter_fail']}")
    print(f"  Final kept: {stats['kept']}")
    
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter questions based on human annotations and update resolution dates"
    )
    parser.add_argument(
        '--file1',
        type=str,
        default='/fast/nchandak/forecasting/evals/freeform/manual/news5-retrieval_1000/fix/grok-4.1-fast:online_eval_size_1000_generations_5_datefiltered.jsonl',
        help='Path to file1 (evaluated questions with idx and correct resolution_date)'
    )
    parser.add_argument(
        '--file2',
        type=str,
        default='/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_manualFilter.jsonl',
        help='Path to file2 (questions with qid and human_filter annotations)'
    )
    parser.add_argument(
        '--file3',
        type=str,
        default='/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_1000_30.jsonl',
        help='Path to file3 (original questions with qid and all data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/fast/nchandak/forecasting/newsdata/testset/o4-mini-high_news5-retrieval_humanFiltered_updated.jsonl',
        help='Path to output JSONL file'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Step 1: Loading files")
    print("=" * 80)
    
    file1_data = load_jsonl(args.file1)
    file2_data = load_jsonl(args.file2)
    file3_data = load_jsonl(args.file3)
    
    print("\n" + "=" * 80)
    print("Step 2: Creating idx to qid mapping")
    print("=" * 80)
    idx_to_qid = create_idx_to_qid_mapping(file1_data, file3_data)
    
    print("\n" + "=" * 80)
    print("Step 3: Extracting resolution dates from file1")
    print("=" * 80)
    idx_to_resolution_date = get_resolution_dates_from_file1(file1_data)
    print(f"  Extracted {len(idx_to_resolution_date)} resolution dates")
    
    print("\n" + "=" * 80)
    print("Step 4: Extracting human_filter annotations from file2")
    print("=" * 80)
    qid_to_human_filter = get_human_filter_mapping(file2_data)
    print(f"  Found {len(qid_to_human_filter)} human annotations")
    
    print("\n" + "=" * 80)
    print("Step 5: Filtering and updating questions")
    print("=" * 80)
    filtered_data = filter_and_update_questions(
        file3_data,
        idx_to_qid,
        idx_to_resolution_date,
        qid_to_human_filter
    )
    
    print("\n" + "=" * 80)
    print("Step 6: Filtering by cutoff date (>= 2025-05-01)")
    print("=" * 80)
    cutoff_date = "2025-05-01"
    before_filter = len(filtered_data)
    filtered_data = [
        entry for entry in filtered_data
        if entry.get('resolution_date', '') >= cutoff_date
    ]
    print(f"  Before cutoff filter: {before_filter}")
    print(f"  After cutoff filter: {len(filtered_data)}")
    print(f"  Removed: {before_filter - len(filtered_data)}")
    
    print("\n" + "=" * 80)
    print("Step 7: Saving results")
    print("=" * 80)
    save_jsonl(filtered_data, args.output)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nOutput file: {args.output}")
    print(f"Total questions: {len(filtered_data)}")


if __name__ == "__main__":
    main()
