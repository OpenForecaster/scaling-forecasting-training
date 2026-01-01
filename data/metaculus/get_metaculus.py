#!/usr/bin/env python3
"""
Metaculus API v2 Question Fetcher (Enhanced)

Purpose:
    Enhanced version of Metaculus question fetcher with date range filtering and better API handling.
    Fetches questions from Metaculus API v2.0 with custom date filters and status options.

Main Features:
    - Date range filtering (from_date, to_date)
    - Status filtering (open, closed, resolved)
    - Automatic pagination and rate limiting
    - Enhanced error handling and retries

Main Functions:
    - fetch_questions_api2(): Main fetching function with API v2
    - process_question(): Converts API response to standardized format
    - parse_date_input(): Parses date strings in 'Month DD, YYYY' format

Usage:
    python get_metaculus.py --from_date "May 01, 2025" --to_date "Nov 01, 2025" --num 1000
    python get_metaculus.py --token YOUR_TOKEN --status resolved --output questions.json
"""

import requests
import json
import argparse
import datetime as dt
import time
from pathlib import Path
from tqdm import tqdm


def normalize_date_string(date_str):
    """
    Normalize a date string by removing milliseconds and ensuring a trailing 'Z'.
    Returns a datetime object or None on error.
    """
    try:
        if date_str is None:
            return None
        if "." in date_str:
            date_str = date_str.split(".")[0]
        if not date_str.endswith("Z"):
            date_str += "Z"
        return dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Error normalizing date string: {date_str}. Error: {e}")
        return None


def parse_date_input(date_str):
    """
    Parse date input in format 'Month DD, YYYY' (e.g., 'May 01, 2025')
    Returns ISO format string: 'YYYY-MM-DDTHH:MM:SSZ'
    """
    try:
        # Parse the date string
        dt_obj = dt.datetime.strptime(date_str, "%B %d, %Y")
        # Return in ISO format with Z suffix
        return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        # Try alternative format
        try:
            dt_obj = dt.datetime.strptime(date_str, "%b %d, %Y")
            return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Expected format: 'Month DD, YYYY' (e.g., 'May 01, 2025')")


def process_question(question):
    """
    Process an individual question dictionary from API2 into a standard format 
    matching metaculus_resolved_filtered.json format.
    
    :param question: The question data from the API2.
    :return: A dictionary with standardized question data.
    """
    # Use actual_resolve_time if available; otherwise, scheduled_resolve_time or resolve_time.
    resolution_date = (
        question.get("actual_resolve_time") or 
        question.get("scheduled_resolve_time") or 
        question.get("resolve_time")
    )
    
    # Build a URL. If the question has a slug, use it; otherwise fall back on the ID.
    slug = question.get("slug")
    if slug:
        url = f"https://www.metaculus.com/questions/{slug}"
    else:
        url = f"https://www.metaculus.com/questions/{question.get('id')}"
    
    # Get question type from possibilities or type field
    question_type = question.get("type")
    if not question_type:
        question_type = question.get("possibilities", {}).get("type")
    
    question_info = {
        "id": question.get("id"),
        "title": question.get("title"),
        "body": question.get("description"),
        "question_type": question_type,
        "resolution_date": resolution_date,  # ISO string or None
        "url": url,
        "data_source": "metaculus",
        "created_date": question.get("created_at") or question.get("created_time") or question.get("publish_time"),
        "metadata": {
            "published_at": question.get("publish_time"),
            "open_time": question.get("open_time"),
            "scheduled_resolve_time": question.get("scheduled_resolve_time"),
            "actual_resolve_time": question.get("actual_resolve_time") or question.get("resolve_time"),
            "scheduled_close_time": question.get("scheduled_close_time"),
            "actual_close_time": question.get("actual_close_time") or question.get("close_time"),
            "nr_forecasters": question.get("nr_forecasters"),
            "possibilities": question.get("possibilities"),
            "options": question.get("options"),
            "resolution_criteria": question.get("resolution_criteria"),
            "fine_print": question.get("fine_print"),
            "post_id": question.get("post_id"),
        },
        "resolution": question.get("resolution")
    }
    return question_info


def fetch_questions(api_url, token, num_questions, open_time=None, close_time=None, status=None, question_type=None, start_offset=0, resolved_after=None):
    """
    Fetch questions from the Metaculus API2 using the questions endpoint.
    
    :param api_url: Base URL of the API (e.g., "https://www.metaculus.com/api2")
    :param token: API token for authentication.
    :param num_questions: Maximum number of questions to fetch.
    :param open_time: Filter by open_time >= this date (ISO format string).
    :param close_time: Filter by close_time <= this date (ISO format string).
    :param status: Filter by status ('open', 'closed', 'resolved').
    :param question_type: Filter by question type (e.g., 'binary', 'numeric', 'multiple_choice').
    :param start_offset: Starting offset for pagination (useful to skip recent unresolved questions).
    :param resolved_after: Filter by questions resolved or closed after this date (ISO format string).
    :return: List of question dictionaries.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    
    questions_info = []
    page_size = 100
    offset = start_offset
    seen_ids = set()
    
    # Base query parameters
    params = {
        "limit": page_size,
    }
    
    # Add date filters - API2 uses these parameter names
    if open_time:
        params["open_time__gt"] = open_time
    if close_time:
        params["close_time__lt"] = close_time
    
    # Add status filter - API2 might use different parameter names
    # Try without status filter first, filter client-side if needed
    # Note: API2 might not support all filters, so we'll filter client-side
    
    # Don't add question_type to params - filter client-side instead
    
    while len(questions_info) < num_questions:
        print(f"Fetching offset {offset}; we have {len(questions_info)} of desired {num_questions} questions")
        
        params["offset"] = offset
        
        try:
            response = requests.get(f"{api_url}/questions", headers=headers, params=params, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                print("Rate limited. Waiting 60 seconds before retrying...")
                time.sleep(60)
                continue
            
            if response.status_code != 200:
                print(f"Warning: API request failed with status {response.status_code}")
                # Print response text for debugging
                try:
                    error_text = response.text[:500]  # First 500 chars
                    print(f"Error response: {error_text}")
                except:
                    pass
                if response.status_code >= 500:
                    print("Server error. Waiting 30 seconds before retrying...")
                    time.sleep(30)
                    continue
                elif response.status_code == 400:
                    # Bad request - might be invalid parameters
                    print("Bad request. Trying without some filters...")
                    # Remove potentially problematic filters
                    if "open_time__gt" in params:
                        del params["open_time__gt"]
                    if "close_time__lt" in params:
                        del params["close_time__lt"]
                    continue
                else:
                    break
            
            data = response.json()
            results = data.get("results", [])
            
            if len(results) == 0:
                print("No more results available")
                break
            
            filtered_counts = {"status": 0, "type": 0, "resolution": 0, "duplicate": 0, "added": 0}
            
            for question in tqdm(results, desc=f"Processing page {offset // page_size + 1}"):
                # Get status - API2 uses 'status' field
                question_status = question.get("status")
                resolved_bool = question.get("resolved", False)
                
                # Filter by status
                if status:
                    status_value = "closed" if status == "close" else status
                    # For resolved, check both status field and resolved boolean
                    if status == "resolved":
                        if question_status != "resolved" or not resolved_bool:
                            filtered_counts["status"] += 1
                            continue
                    else:
                        if question_status != status_value:
                            filtered_counts["status"] += 1
                            continue
                
                # For questions that pass status filter, fetch full details to get type and resolution
                qid = question.get("id")
                if qid and qid not in seen_ids:
                    try:
                        detail_response = requests.get(f"{api_url}/questions/{qid}/", headers=headers, timeout=10)
                        if detail_response.status_code == 200:
                            question_detail = detail_response.json()
                        else:
                            # If detail fetch fails, use the list data
                            question_detail = question
                    except:
                        question_detail = question
                else:
                    continue
                
                # Get question type from detail
                q_type = question_detail.get("type")
                if not q_type and "possibilities" in question_detail:
                    q_type = question_detail.get("possibilities", {}).get("type")
                
                # Filter by question type
                if question_type and q_type != question_type:
                    filtered_counts["type"] += 1
                    continue
                
                # Get resolution
                resolution = question_detail.get("resolution")
                
                # Filter out questions with resolution "annulled" only
                # Note: We allow None/null resolutions as some resolved questions may not have this field populated
                if resolution == "annulled":
                    filtered_counts["resolution"] += 1
                    continue
                
                # Filter by resolved_after date (check both resolve_time and close_time)
                if resolved_after:
                    resolve_time = (
                        question_detail.get("actual_resolve_time") or 
                        question_detail.get("resolve_time")
                    )
                    close_time_val = (
                        question_detail.get("actual_close_time") or 
                        question_detail.get("close_time")
                    )
                    
                    # Parse the times
                    resolve_dt = normalize_date_string(resolve_time) if resolve_time else None
                    close_dt = normalize_date_string(close_time_val) if close_time_val else None
                    resolved_after_dt = normalize_date_string(resolved_after)
                    
                    # Question must have resolved or closed after the specified date
                    if resolved_after_dt:
                        has_valid_time = False
                        if resolve_dt and resolve_dt >= resolved_after_dt:
                            has_valid_time = True
                        if close_dt and close_dt >= resolved_after_dt:
                            has_valid_time = True
                        
                        if not has_valid_time:
                            filtered_counts["resolution"] += 1
                            continue
                
                # Process the question with full details
                q = process_question(question_detail)
                
                # Skip duplicates
                if q["id"] not in seen_ids:
                    questions_info.append(q)
                    seen_ids.add(q["id"])
                    filtered_counts["added"] += 1
                    # Small delay after each detail fetch
                    time.sleep(0.1)
                else:
                    filtered_counts["duplicate"] += 1
                
                if len(questions_info) >= num_questions:
                    break
            
            print(f"  Filtered this page: {filtered_counts} | Total collected: {len(questions_info)}")
            
            if len(questions_info) >= num_questions:
                break
            
            # Check if there are more results
            if not data.get("next") and len(results) < page_size:
                print("No more pages available")
                break
            
            offset += page_size
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            print("Waiting 10 seconds before retrying...")
            time.sleep(10)
            continue
    
    print(f"Fetched {len(questions_info)} questions")
    return questions_info[:num_questions]


if __name__ == "__main__":
    # Load default token from file
    from pathlib import Path
    token_file = Path(__file__).parent / "metaculus_token.txt"
    if token_file.exists():
        default_token = token_file.read_text().strip()
    else:
        default_token = ""
        print(f"Warning: metaculus_token.txt not found at {token_file}")
    
    parser = argparse.ArgumentParser(
        description="Scrape questions from Metaculus API"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=default_token,
        help="API token for Metaculus"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of questions to scrape"
    )
    parser.add_argument(
        "--open_time",
        type=str,
        default="May 01, 2000",
        help="Open time filter (format: 'Month DD, YYYY', e.g., 'May 01, 2025')"
    )
    parser.add_argument(
        "--close_time",
        type=str,
        default=None,
        help="Close time filter (format: 'Month DD, YYYY', e.g., 'May 01, 2025')"
    )
    parser.add_argument(
        "--resolved_after",
        type=str,
        default="May 01, 2025",
        help="Filter questions that resolved or closed after this date (format: 'Month DD, YYYY', e.g., 'May 01, 2025'). Leave empty for no filter."
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["open", "close", "resolved"],
        default="resolved",
        help="Filter by question status (open/close/resolved)"
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default=None,
        help="Filter by question type (e.g., 'binary', 'numeric', 'multiple_choice')"
    )
    parser.add_argument(
        "--start_offset",
        type=int,
        default=0,
        help="Starting offset for pagination (default: 0 for recently resolved questions, use higher values like 5000+ for older questions)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metaculus_resolved_new123.json",
        help="Output JSON file path (default: prints to stdout)"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="https://www.metaculus.com/api2",
        help="Base URL for the Metaculus API"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    open_time_iso = None
    close_time_iso = None
    resolved_after_iso = None
    
    try:
        if args.open_time and args.open_time.strip():
            open_time_iso = parse_date_input(args.open_time)
            print(f"Filtering by open_time >= {open_time_iso}")
        else:
            open_time_iso = None
        
        if args.close_time and args.close_time.strip():
            close_time_iso = parse_date_input(args.close_time)
            print(f"Filtering by close_time <= {close_time_iso}")
        else:
            close_time_iso = None
        
        if args.resolved_after and args.resolved_after.strip():
            resolved_after_iso = parse_date_input(args.resolved_after)
            print(f"Filtering by resolved_after >= {resolved_after_iso}")
        else:
            resolved_after_iso = None
        
        if args.status:
            print(f"Filtering by status: {args.status}")
        
        if args.question_type:
            print(f"Filtering by question_type: {args.question_type}")
    except ValueError as e:
        print(f"Error parsing date: {e}")
        exit(1)
    
    try:
        data = fetch_questions(
            api_url=args.api_url,
            token=args.token,
            num_questions=args.num,
            open_time=open_time_iso,
            close_time=close_time_iso,
            status=args.status,
            question_type=args.question_type,
            start_offset=args.start_offset,
            resolved_after=resolved_after_iso
        )
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w') as f:
                json.dump(data, f, indent=4)
            print(f"\nSaved {len(data)} questions to {output_path}")
        else:
            print(f"\nFetched {len(data)} questions:")
            print(json.dumps(data, indent=4))
            
    except Exception as e:
        print(f"Error: {e}")
        raise e

