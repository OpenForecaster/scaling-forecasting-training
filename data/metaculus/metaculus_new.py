#!/usr/bin/env python3
"""
Metaculus API Question Fetcher

Purpose:
    Fetches forecasting questions from Metaculus API v2.0 and converts them to standardized format.
    Supports filtering by date ranges, status, and question types.

Main Functions:
    - fetch_posts_with_questions(): Fetches questions from Metaculus API with pagination
    - process_question(): Converts Metaculus question format to standardized schema
    - normalize_date_string(): Normalizes various date formats to ISO8601

API Documentation:
    https://www.metaculus.com/api/

Output Format:
    Standardized JSON with fields: id, title, body, question_type, resolution_date, url,
    data_source, created_date, metadata, resolution

Usage:
    python metaculus_new.py --token YOUR_TOKEN --num 500 --output metaculus_questions.json
    python metaculus_new.py --params '{"open_time__gt": "2020-01-01T00:00:00Z"}' --num 1000
"""

import requests
import json
import argparse
import datetime as dt
from pathlib import Path
from tqdm import tqdm
import sys

def dump_questions(questions, output_path, append=False):
    """
    Dump questions to a JSON file, either overwriting or appending.
    """
    mode = 'a' if append else 'w'
    if append and output_path.exists():
        # If appending, read existing data first
        with output_path.open('r') as f:
            existing_data = json.load(f)
        questions = existing_data + questions
    
    with output_path.open(mode) as f:
        for _ in tqdm(range(1), desc="Saving questions"):
            json.dump(questions, f, indent=4)
    print(f"Saved {len(questions)} questions to {output_path}")

def call_api(url, headers=None, params=None):
    """
    Make an API call with basic error handling.
    """
    try:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=1)
        except requests.exceptions.Timeout:
            print(f"Warning: API request timed out for URL: {url}")
            response = type('Response', (), {'status_code': 408, 'text': 'Request timed out'})()
        
        if response.status_code != 200:
            print(f"Warning: API request failed: {response.status_code} - {response.text}")
            return None
        return response
    except requests.exceptions.RequestException as e:
        print(f"Warning: API request error: {e}")
        return None

def normalize_date_string(date_str):
    """
    Normalize a date string by removing milliseconds and ensuring a trailing 'Z'.
    Returns a datetime object or None on error.
    """
    try:
        if date_str is None:
            return None
        if "." in date_str:
            # Remove any fractional seconds.
            date_str = date_str.split(".")[0]
        if not date_str.endswith("Z"):
            date_str += "Z"
        return dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Error normalizing date string: {date_str}. Error: {e}")
        return None

def process_question(question, parent_post, group=None):
    """
    Process an individual question dictionary (from a post or from a group-of-questions)
    into a standard format.
    
    :param question: The question data from the API.
    :param parent_post: The post that contains the question (for extra info such as published_at).
    :param group: (Optional) If this question is part of a group, pass the group dict.
    :return: A dictionary with standardized question data.
    """
    # Use actual_resolve_time if available; otherwise, scheduled_resolve_time.
    resolution_date = question.get("actual_resolve_time") or question.get("scheduled_resolve_time")
    
    # Build a URL. If the question has a slug, use it; otherwise fall back on the ID.
    slug = question.get("slug")
    if slug:
        url = f"https://www.metaculus.com/questions/{slug}"
    else:
        url = f"https://www.metaculus.com/questions/{question.get('id')}"
    
    question_info = {
        "id": question.get("id"),
        "title": question.get("title"),
        "body": question.get("description"),
        "question_type": question.get("type"),
        "resolution_date": resolution_date,  # ISO string or None
        "url": url,
        "data_source": "metaculus",
        "created_date": question.get("created_at"),
        "metadata": {
            "published_at": parent_post.get("published_at"),
            "open_time": question.get("open_time"),
            "scheduled_resolve_time": question.get("scheduled_resolve_time"),
            "actual_resolve_time": question.get("actual_resolve_time"),
            "scheduled_close_time": question.get("scheduled_close_time"),
            "actual_close_time": question.get("actual_close_time"),
            "nr_forecasters": parent_post.get("nr_forecasters"),
            "possibilities": question.get("possibilities"),
            "options": question.get("options"),
            "resolution_criteria": question.get("resolution_criteria"),
            "fine_print": question.get("fine_print"),
            "post_id": question.get("post_id"),
        },
        "resolution": question.get("resolution")
    }
    return question_info

def fetch_posts_with_questions(api_url, output_path, token=None, query_params={}, total_questions=500):
    """
    Fetch posts from the Metaculus API v2.0 and extract questions.
    You can supply extra query parameters (e.g., filtering by open_time__gt) via a dictionary.
    
    :param api_url: Base URL of the API (e.g., "https://www.metaculus.com/api")
    :param output_path: Path object for the output JSON file
    :param token: API token for authentication.
    :param query_params: Dictionary of query parameters to filter the posts feed.
    :param total_questions: Maximum number of questions to fetch.
    :return: List of question dictionaries.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Token {token}"
    
    questions_info = []
    next_url = f"{api_url}/posts/"
    first_request = True

    while next_url and len(questions_info) < total_questions:
        print(next_url)
        if first_request:
            response = call_api(next_url, headers, params=query_params)
            first_request = False
        else:
            response = call_api(next_url, headers)
        
        if response is None:
            print("Skipping this batch due to API error")
            continue
        
        data = response.json()
        results = data.get("results", [])
        if len(results):
            for post in tqdm(results, desc="Processing posts"):
                # If the post contains an individual question.
                if "question" in post and post["question"]:
                    q = process_question(post["question"], parent_post=post)
                    questions_info.append(q)
                # If the post is a group-of-questions, iterate over its questions.
                if "group_of_questions" in post and post["group_of_questions"]:
                    group_info = post["group_of_questions"]
                    for subq in group_info.get("questions", []):
                        q = process_question(subq, parent_post=post, group=group_info)
                        questions_info.append(q)
        
        else:
            break
        
        if len(questions_info) >= total_questions:
            break
        
        next_url = data.get("next")
    
    # Final dump of questions
    dump_questions(questions_info, output_path)
    print(f"Fetched {len(questions_info)} questions.")
    return questions_info[:total_questions]

if __name__ == "__main__":
    # Load default token from file
    from pathlib import Path
    token_file = Path(__file__).parent / "metaculus_token.txt"
    if token_file.exists():
        default_token = token_file.read_text().strip()
    else:
        default_token = ""
        print(f"Warning: metaculus_token.txt not found at {token_file}")
    
    parser = argparse.ArgumentParser(description="Fetch questions from Metaculus API v2.0")
    parser.add_argument("--token", type=str, default=default_token, help="API token for Metaculus")
    parser.add_argument(
        "--params",
        type=str,
        help=("JSON string of query parameters for the API. "
              "For example: '{\"open_time__gt\": \"2020-01-01T00:00:00Z\", \"statuses\": [\"open\"]}'")
    )
    parser.add_argument("--num", type=int, default=5, help="Number of questions to fetch")
    parser.add_argument("--output", type=str, default="metaculus_questions.json", help="Output JSON file")
    parser.add_argument("--api_url", type=str, default="https://www.metaculus.com/api", help="Base URL for the Metaculus API")
    
    args = parser.parse_args()
    
    # Parse query parameters from JSON string if provided.
    query_params = {}
    if args.params:
        try:
            query_params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"Error parsing params JSON: {e}")
            exit(1)
    
    try:
        output_path = Path(args.output)
        data = fetch_posts_with_questions(
            api_url=args.api_url,
            output_path=output_path,
            token=args.token,
            query_params=query_params,
            total_questions=args.num
        )
        print("Total questions fetched:", len(data))
    except Exception as e:
        print(f"Error: {e}")
        raise e
