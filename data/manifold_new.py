#!/usr/bin/env python3
"""
Manifold Markets Data Processor

Purpose:
    Converts Manifold Markets prediction market data to standardized format compatible with Metaculus.
    Processes BINARY, MULTIPLE_CHOICE, and NUMBER type questions from Manifold's data dumps.

Main Functions:
    - process_manifold_question(): Converts single Manifold question to standard format
    - ms_to_iso(): Converts millisecond timestamps to ISO8601
    - extract_text_from_description(): Extracts text from rich text format

Input:
    - Manifold data dumps (JSON or JSONL format) from https://docs.manifold.markets/api

Output:
    - Standardized JSON file with processed questions in Metaculus-compatible format

Usage:
    python manifold_new.py --input_path data/manifold-contracts.json --output_path data/manifold_resolved.json
"""

import json
from pathlib import Path
from datetime import datetime
import argparse
def ms_to_iso(ms):
    """Convert milliseconds since epoch to ISO8601 string (UTC)."""
    try:
        return datetime.utcfromtimestamp(ms / 1000).isoformat() + "Z"
    except Exception as e:
        print(f"Error converting timestamp {ms}: {e}")
        return None

def extract_text_from_description(desc):
    """
    If description is a dict (rich text format), extract text from its content.
    Otherwise, return it as a string.
    """
    if isinstance(desc, dict) and "content" in desc:
        texts = []
        def recurse(content):
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    if "content" in item:
                        recurse(item["content"])
        recurse(desc.get("content", []))
        return "\n".join(texts)
    elif isinstance(desc, str):
        return desc
    else:
        return ""

def process_manifold_question(q):
    """
    Convert a single manifold question dict to the metaculus question_info format.
    Only process if the question is resolved and its outcomeType is one of:
    BINARY, MULTIPLE_CHOICE, or NUMBER.
    """
    # If input is a JSONL entry with full_market, extract the full_market
    if "full_market" in q:
        q = q.get("full_market", {})
    
    if not q.get("isResolved"):
        return None

    outcome = q.get("outcomeType")
    
    if outcome not in ["BINARY", "MULTIPLE_CHOICE", "NUMBER"]:
        return None

    # Filter out questions with resolution 'MKT' or 'CANCEL'
    resolution = q.get("resolution")
    if resolution in ["MKT", "CANCEL"]:
        return None

    # Map manifold outcomeType to our question_type format.
    outcome_map = {"BINARY": "binary",
                   "MULTIPLE_CHOICE": "multiple_choice",
                   "NUMBER": "number"}
    question_type = outcome_map.get(outcome, "unknown")

    # Convert timestamps (assume they are in milliseconds).
    created_date = ms_to_iso(q.get("createdTime")) if q.get("createdTime") else None
    resolution_date = ms_to_iso(q.get("closeTime")) if q.get("closeTime") else None

    # Title and body: use 'question' as title and 'description' (processed) as body.
    title = q.get("question", "")
    desc = q.get("description", "")
    body = extract_text_from_description(desc) if desc else ""

    # Build URL using slug if available; otherwise, use id.
    slug = q.get("slug")
    if slug:
        url = f"https://manifold.markets/{slug}"
    else:
        url = f"https://manifold.markets/contract/{q.get('id')}"

    # Build metadata using available fields.
    metadata = {
        "published_at": created_date,
        "open_time": created_date,
        "scheduled_resolve_time": resolution_date,
        "actual_resolve_time": resolution_date,
        "scheduled_close_time": resolution_date,
        "actual_close_time": resolution_date,
        "nr_forecasters": q.get("uniqueBettorCount"),  # if available
        "possibilities": {},
        "options": None,  # will be set for multiple_choice below
        "resolution_criteria": q.get("resolution_criteria", ""),
        "fine_print": q.get("fine_print", ""),
        "post_id": q.get("id"),
        # Additional manifold fields:
        "mechanism": q.get("mechanism"),
        "volume": q.get("volume"),
        "creatorId": q.get("creatorId"),
        "creatorName": q.get("creatorName"),
        "marketTier": q.get("marketTier"),
        "outcomeType": outcome
    }

    # For binary questions, lower-case the keys in the pool.
    if outcome == "BINARY":
        pool = q.get("pool", {})
        metadata["possibilities"] = {k.lower(): v for k, v in pool.items()}
        # Lowercase the resolution for binary data.
        resolution = resolution.lower() if resolution in ["YES", "NO"] else resolution

    # For number questions, include min and max if available.
    if outcome == "NUMBER":
        metadata["min"] = q.get("min")
        metadata["max"] = q.get("max")

    # For multiple_choice, extract options (only the text) from the 'answers' field.
    if outcome == "MULTIPLE_CHOICE":
        answers = q.get("answers", [])
        options = [ans.get("text", "") for ans in answers if ans.get("text")]
        metadata["options"] = options

        # Map the resolution from an answer option id to its text.
        resolution_option_id = q.get("resolution")
        mapped_resolution = None
        for ans in answers:
            if ans.get("id") == resolution_option_id:
                mapped_resolution = ans.get("text", "")
                break
        if not mapped_resolution:
            # If no answer option matches the resolution id, skip this question.
            return None
        final_resolution = mapped_resolution
    else:
        final_resolution = resolution

    processed = {
        "id": q.get("id"),
        "title": title,
        "body": body,
        "question_type": question_type,
        "resolution_date": resolution_date,
        "url": url,
        "data_source": "manifold",
        "created_date": created_date,
        "metadata": metadata,
        "resolution": final_resolution
    }
    return processed
def main():
    parser = argparse.ArgumentParser(description='Process Manifold questions')
    parser.add_argument('--input_path', type=str, default="data/manifold-contracts-20240706.json",
                        help='Path to the input JSON file')
    parser.add_argument('--output_path', type=str, default="data/manifold_resolved.json",
                        help='Path to save the processed questions')
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # Check if the input file is JSON or JSONL based on extension
    is_jsonl = input_path.suffix.lower() == ".jsonl"
    
    processed_questions = []
    
    if is_jsonl:
        # Process JSONL file line by line
        with input_path.open("r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        q = json.loads(line)
                        processed = process_manifold_question(q)
                        if processed:
                            processed_questions.append(processed)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON line: {e}")
    else:
        # Process regular JSON file
        with input_path.open("r") as f:
            data = json.load(f)
        
        # Assuming the manifold dump is a list of contract dicts:
        for q in data:
            processed = process_manifold_question(q)
            if processed:
                processed_questions.append(processed)
    
    with output_path.open("w") as f:
        json.dump(processed_questions, f, indent=4)
    
    print(f"Saved {len(processed_questions)} questions to {output_path}")

if __name__ == "__main__":
    main()
