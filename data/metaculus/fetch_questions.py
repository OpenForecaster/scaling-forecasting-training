#!/usr/bin/env python3
import requests
import json
import argparse
import datetime as dt
import time
import sys
import re
from dateutil import parser as date_parser

# Constants
API_BASE_URL = "https://www.metaculus.com/api2"
LEGACY_API_BASE = "https://www.metaculus.com/api"

# Load default token from file
from pathlib import Path as PathLib
token_file = PathLib(__file__).parent / "metaculus_token.txt"
if token_file.exists():
    DEFAULT_TOKEN = token_file.read_text().strip()
else:
    DEFAULT_TOKEN = ""
    print(f"Warning: metaculus_token.txt not found at {token_file}")

POST_DETAIL_CACHE = {}
URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

def parse_date(date_str):
    """Parses a date string into a timezone-aware datetime object (UTC)."""
    if not date_str:
        return None
    try:
        dt_obj = date_parser.parse(date_str)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
        else:
            dt_obj = dt_obj.astimezone(dt.timezone.utc)
        return dt_obj
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}", file=sys.stderr)
        return None


def first_non_none(*values):
    """Return the first non-None value."""
    for value in values:
        if value is not None:
            return value
    return None


def extract_urls_from_text(*texts):
    """
    Extract unique URLs from any textual fields, preserving order of appearance.
    """
    seen = set()
    urls = []
    for text in texts:
        if not text:
            continue
        for match in URL_PATTERN.findall(text):
            cleaned = match.rstrip('.,)"\'')
            if cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
    return urls


def fetch_post_question(post_id, headers):
    """
    Fetch the richer question payload (with description, resolution criteria, etc.)
    from the legacy /api/posts endpoint.
    """
    if not post_id:
        return None

    if post_id in POST_DETAIL_CACHE:
        return POST_DETAIL_CACHE[post_id]

    legacy_headers = {"User-Agent": headers.get("User-Agent", "Mozilla/5.0")}
    if "Authorization" in headers:
        legacy_headers["Authorization"] = headers["Authorization"]

    try:
        response = requests.get(
            f"{LEGACY_API_BASE}/posts/{post_id}/",
            headers=legacy_headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            question_payload = data.get("question") or {}
            POST_DETAIL_CACHE[post_id] = question_payload
            return question_payload
        else:
            print(f"Warning: legacy post fetch failed for {post_id}: {response.status_code}")
    except requests.exceptions.RequestException as exc:
        print(f"Warning: error fetching legacy post {post_id}: {exc}")

    POST_DETAIL_CACHE[post_id] = None
    return None


def fetch_prediction_history(question_id, headers):
    """
    Fetch prediction history for a question to get the final community prediction.
    Returns the most recent community prediction value, or None if not available.
    """
    if not question_id:
        return None

    try:
        # Try the prediction history endpoint
        response = requests.get(
            f"{API_BASE_URL}/questions/{question_id}/prediction_history/",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            # Prediction history is typically a list of predictions over time
            # Get the most recent (last) prediction
            if isinstance(data, list) and len(data) > 0:
                # Get the last entry (most recent)
                latest = data[-1]
                # Community prediction might be in 'cp' or 'community_prediction' field
                return latest.get("cp") or latest.get("community_prediction")
            elif isinstance(data, dict):
                # Sometimes it's a dict with a 'results' key
                results = data.get("results", [])
                if results:
                    latest = results[-1]
                    return latest.get("cp") or latest.get("community_prediction")
    except requests.exceptions.RequestException as exc:
        # Silently fail - prediction history might not be available for all questions
        pass

    return None


def transform_question(question_detail, legacy_question=None, prediction_history_value=None):
    """
    Transform the Metaculus question detail payload into the filtered JSON structure.
    
    Args:
        question_detail: The main question detail from API
        legacy_question: Optional legacy question data from posts endpoint
        prediction_history_value: Optional community prediction from prediction history endpoint
    """
    nested_question = question_detail.get("question") or {}
    legacy_question = legacy_question or {}

    resolution_date = first_non_none(
        question_detail.get("actual_resolve_time"),
        question_detail.get("scheduled_resolve_time"),
        question_detail.get("resolve_time"),
        nested_question.get("actual_resolve_time"),
        nested_question.get("scheduled_resolve_time"),
        nested_question.get("resolve_time"),
    )

    slug = first_non_none(question_detail.get("slug"), nested_question.get("slug"))
    if slug:
        url = f"https://www.metaculus.com/questions/{slug}"
    else:
        url = f"https://www.metaculus.com/questions/{first_non_none(question_detail.get('id'), nested_question.get('id'))}"

    def possibility_type(payload):
        possibilities = payload.get("possibilities")
        if isinstance(possibilities, dict):
            return possibilities.get("type")
        return None

    question_type = first_non_none(
        question_detail.get("type"),
        possibility_type(question_detail),
        nested_question.get("type"),
        possibility_type(nested_question),
        legacy_question.get("type"),
        possibility_type(legacy_question),
    )

    body_text = first_non_none(
        question_detail.get("description"),
        question_detail.get("body"),
        nested_question.get("description"),
        nested_question.get("body"),
        legacy_question.get("description"),
        legacy_question.get("body"),
    ) or ""

    resolution_criteria_text = first_non_none(
        question_detail.get("resolution_criteria"),
        nested_question.get("resolution_criteria"),
        legacy_question.get("resolution_criteria"),
    ) or ""

    fine_print_text = first_non_none(
        question_detail.get("fine_print"),
        nested_question.get("fine_print"),
        legacy_question.get("fine_print"),
    ) or ""

    possibilities_value = first_non_none(
        question_detail.get("possibilities"),
        nested_question.get("possibilities"),
        legacy_question.get("possibilities"),
    ) or {}

    options_value = first_non_none(
        question_detail.get("options"),
        nested_question.get("options"),
        legacy_question.get("options"),
    )
    if options_value is None:
        options_value = []

    extracted_urls = extract_urls_from_text(body_text, resolution_criteria_text, fine_print_text)

    # Extract community prediction (market prediction)
    # For binary questions, this is typically a probability (0-1) for "Yes"
    # The community prediction is stored in question.aggregations.recency_weighted.latest.centers[0]
    community_prediction = None
    
    # First, try to get from aggregations (most reliable location)
    aggregations = nested_question.get("aggregations") or question_detail.get("aggregations")
    if aggregations and isinstance(aggregations, dict):
        recency_weighted = aggregations.get("recency_weighted")
        if recency_weighted and isinstance(recency_weighted, dict):
            latest = recency_weighted.get("latest")
            if latest and isinstance(latest, dict):
                centers = latest.get("centers")
                if centers and isinstance(centers, list) and len(centers) > 0:
                    # For binary questions, centers[0] is the probability of "Yes"
                    community_prediction = centers[0]
    
    # Fallback to other possible locations
    if community_prediction is None:
        community_prediction = first_non_none(
            prediction_history_value,  # From prediction history endpoint if available
            question_detail.get("community_prediction"),
            question_detail.get("cp"),
            nested_question.get("community_prediction"),
            nested_question.get("cp"),
            legacy_question.get("community_prediction"),
            legacy_question.get("cp"),
        )
    
    # Also check if it's nested in a predictions or forecast structure
    if community_prediction is None:
        predictions_data = first_non_none(
            question_detail.get("predictions"),
            nested_question.get("predictions"),
            legacy_question.get("predictions"),
        )
        if isinstance(predictions_data, dict):
            community_prediction = predictions_data.get("community_prediction") or predictions_data.get("cp")
    
    # Check forecast_history or similar structures
    if community_prediction is None:
        forecast_data = first_non_none(
            question_detail.get("forecast"),
            nested_question.get("forecast"),
            legacy_question.get("forecast"),
        )
        if isinstance(forecast_data, dict):
            community_prediction = forecast_data.get("community_prediction") or forecast_data.get("cp")

    standardized = {
        "id": first_non_none(question_detail.get("id"), nested_question.get("id")),
        "title": first_non_none(question_detail.get("title"), nested_question.get("title")),
        "body": body_text,
        "question_type": question_type,
        "resolution_date": resolution_date,
        "url": url,
        "data_source": "metaculus",
        "created_date": first_non_none(
            question_detail.get("created_at"),
            question_detail.get("created_time"),
            question_detail.get("published_at"),
            question_detail.get("publish_time"),
            nested_question.get("created_at"),
            nested_question.get("created_time"),
            nested_question.get("published_at"),
            nested_question.get("publish_time"),
        ),
        "metadata": {
            "published_at": first_non_none(
                question_detail.get("published_at"),
                question_detail.get("publish_time"),
                nested_question.get("published_at"),
                nested_question.get("publish_time"),
            ),
            "open_time": first_non_none(question_detail.get("open_time"), nested_question.get("open_time")),
            "scheduled_resolve_time": first_non_none(
                question_detail.get("scheduled_resolve_time"),
                nested_question.get("scheduled_resolve_time"),
            ),
            "actual_resolve_time": first_non_none(
                question_detail.get("actual_resolve_time"),
                question_detail.get("resolve_time"),
                nested_question.get("actual_resolve_time"),
                nested_question.get("resolve_time"),
            ),
            "scheduled_close_time": first_non_none(
                question_detail.get("scheduled_close_time"),
                nested_question.get("scheduled_close_time"),
            ),
            "actual_close_time": first_non_none(
                question_detail.get("actual_close_time"),
                question_detail.get("close_time"),
                nested_question.get("actual_close_time"),
                nested_question.get("close_time"),
            ),
            "nr_forecasters": first_non_none(
                question_detail.get("nr_forecasters"),
                nested_question.get("nr_forecasters"),
                legacy_question.get("nr_forecasters"),
            ),
            "possibilities": possibilities_value,
            "options": options_value,
            "resolution_criteria": resolution_criteria_text,
            "fine_print": fine_print_text,
            "post_id": first_non_none(
                question_detail.get("post_id"), nested_question.get("post_id"), legacy_question.get("post_id")
            ),
        },
        "resolution": first_non_none(question_detail.get("resolution"), nested_question.get("resolution")),
        "resolution_criteria": resolution_criteria_text,
        "extracted_urls": extracted_urls,
        "market_prediction": community_prediction,
    }
    return standardized

def fetch_all_resolved_questions(token, resolved_after_dt, output_file, limit):
    """
    Fetches resolved questions from Metaculus API resolved after the given datetime.
    Saves them to the output file in JSONL format.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Token {token}"
    
    # Initial params
    params = {
        "limit": 100,
        "offset": 0,
        # We can filter by resolved status, but API filtering might be limited.
        # We will filter client-side for date to be safe and precise.
    }

    print(f"Fetching questions resolved after {resolved_after_dt}...", file=sys.stderr)
    
    count = 0
    url = f"{API_BASE_URL}/questions/"
    
    with open(output_file, 'w') as f:
        while url:
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                
                for question_summary in results:
                    title = (question_summary.get("title") or "").lower().strip()
                    if title.startswith("will the community prediction"):
                        continue
                    
                    notrequired = "be higher than its market close price on"
                    if notrequired in title:
                        continue
                    
                    # Check if resolved
                    # 'status' field is usually 'resolved' for resolved questions.
                    # Also check 'resolved' boolean if present.
                    is_resolved = False
                    if question_summary.get('status') == 'resolved':
                        is_resolved = True
                    elif question_summary.get('resolved'):
                        is_resolved = True
                        
                    if not is_resolved:
                        continue

                    # Get resolution time
                    # Possible fields: actual_resolve_time, resolve_time, scheduled_resolve_time
                    # We prioritize actual_resolve_time
                    resolve_time_str = (
                        question_summary.get("actual_resolve_time") or 
                        question_summary.get("resolve_time")
                    )
                    
                    if not resolve_time_str:
                        # If resolved but no time, we might skip or include depending on strictness.
                        # Usually resolved questions have a time.
                        continue
                        
                    resolve_time = parse_date(resolve_time_str)
                    
                    # Filter by time
                    if resolved_after_dt and resolve_time and resolve_time < resolved_after_dt:
                        continue

                    # Fetch full details
                    # The list endpoint might not have "every detail" (e.g. prediction history usually requires separate endpoint or full detail fetch).
                    # For "every detail", let's hit the detail endpoint.
                    question_id = question_summary.get('id')
                    if not question_id:
                        continue
                        
                    try:
                        detail_url = f"{API_BASE_URL}/questions/{question_id}/"
                        detail_res = requests.get(detail_url, headers=headers)
                        if detail_res.status_code == 200:
                            full_question = detail_res.json()
                        else:
                            print(f"Failed to fetch details for {question_id}, using summary.", file=sys.stderr)
                            full_question = question_summary
                    except Exception as e:
                        print(f"Error fetching details for {question_id}: {e}", file=sys.stderr)
                        full_question = question_summary

                    # Enrich with legacy body text before standardizing
                    nested_question_detail = full_question.get("question") or {}
                    post_id = first_non_none(
                        full_question.get("post_id"),
                        nested_question_detail.get("post_id"),
                    )
                    legacy_question = fetch_post_question(post_id, headers)

                    # Fetch prediction history to get final community prediction before resolution
                    prediction_history_value = fetch_prediction_history(question_id, headers)

                    # Write to JSONL in standardized format (after filtering for binary yes/no)
                    standardized = transform_question(
                        full_question, 
                        legacy_question=legacy_question,
                        prediction_history_value=prediction_history_value
                    )

                    if standardized.get("question_type") != "binary":
                        continue

                    resolution_value = standardized.get("resolution")
                    if resolution_value is None or str(resolution_value).lower() not in {"yes", "no"}:
                        continue

                    f.write(json.dumps(standardized) + "\n")
                    count += 1
                    if count % 10 == 0 or count == limit:
                        print(f"Collected {count} questions...", end='\r', file=sys.stderr)

                    if count >= limit:
                        break

                # Pagination
                url = data.get('next')
                # The 'next' URL usually includes params, so we clear local params to avoid duplication if requests merges them incorrectly, 
                # but requests usually handles full URLs fine. 
                # However, if 'next' is provided, we use it as the url and clear params.
                if url:
                    params = {}

                if count >= limit:
                    break
                
            except requests.exceptions.RequestException as e:
                print(f"\nRequest failed: {e}", file=sys.stderr)
                time.sleep(5) # Wait a bit before retry or exit? 
                # For simplicity in this script, we might just stop or retry. 
                # Let's retry once then fail.
                # (Implementing simple retry logic would be better but keeping it simple for now unless requested)
                break
            except KeyboardInterrupt:
                print("\nStopping...", file=sys.stderr)
                break
                
    print(f"\nDone. Saved {count} questions to {output_file}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch resolved questions from Metaculus.")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Metaculus API Token")
    parser.add_argument(
        "--time",
        default="2025-10-01",
        help="Filter questions resolved after this time (default: 2025-10-01). Supports flexible date formats.",
    )
    parser.add_argument("--output", default="metaculus_resolved12345.jsonl", help="Output JSONL file path")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of questions to save (default: 10 for quick verification).",
    )
    
    args = parser.parse_args()
    
    resolved_after_dt = parse_date(args.time)

    fetch_all_resolved_questions(args.token, resolved_after_dt, args.output, args.limit)
