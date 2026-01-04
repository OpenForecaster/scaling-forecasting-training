#!/usr/bin/env python3
"""
Metaculus to Standardized Format Converter

Purpose:
    Converts Metaculus JSONL data dumps into standardized binary question format
    for consistent training and evaluation across different platforms.

Main Functions:
    - convert_metaculus_question(): Converts single Metaculus question to standard format
    - _normalize_answer(): Normalizes YES/NO answers to consistent format
    - _clean_string(): Handles None and empty string values

Output Format:
    Standardized JSONL with fields: question, background, resolution_criteria,
    answer, answer_type, date fields, url, data_source

Usage:
    python convert_to_standardized_format.py --input metaculusOct.jsonl --output standardized.jsonl
"""

import argparse
import json
import os
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Metaculus JSONL dumps into the standardized binary question format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source JSONL file (e.g., metaculusOct.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSONL file path for the converted data.",
    )
    return parser.parse_args()


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_answer(answer: Any) -> str:
    text = _clean_string(answer).strip()
    if not text:
        return ""
    normalized = text.lower()
    if normalized in {"yes", "y", "true", "1"}:
        return "Yes"
    if normalized in {"no", "n", "false", "0"}:
        return "No"
    return text


def _normalize_answer_type(answer_type: Any, question_type: Any) -> str:
    candidate = (_clean_string(answer_type) or _clean_string(question_type)).lower()
    if "binary" in candidate:
        return "binary"
    return answer_type or question_type or "binary"


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    question_title = (
        record.get("question_title")
        or record.get("question")
        or record.get("title")
        or ""
    )

    converted = {
        "question_title": question_title,
        "background": record.get("background") or "",
        "resolution_criteria": record.get("resolution_criteria") or "",
        "answer_type": _normalize_answer_type(
            record.get("answer_type"), record.get("question_type")
        ),
        "answer": _normalize_answer(record.get("answer")),
        "resolution_date": record.get("date_resolve_at") or "",
        "question_close_date": record.get("date_close") or "",
        "question_start_date": record.get("date_begin") or "",
        "url": record.get("url") or "",
        "article_maintext": record.get("article_maintext") or "",
        "article_publish_date": record.get("article_publish_date") or "",
        "article_modify_date": record.get("article_modify_date") or "",
        "article_download_date": record.get("article_download_date") or "",
        "data_source": record.get("data_source") or "metaculus",
        "news_source": record.get("news_source")
        or record.get("data_source")
        or "metaculus",
        "resolution": int(record["resolution"])
        if isinstance(record.get("resolution"), (int, bool))
        else (
            1 if str(record.get("resolution")).strip().lower() == "yes" else 0
        ),
    }
    return converted


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = 0
    with open(input_path, "r") as src, open(output_path, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            converted = convert_record(record)
            dst.write(json.dumps(converted, ensure_ascii=True) + "\n")
            total += 1

    print(f"Converted {total} records -> {output_path}")


if __name__ == "__main__":
    main()

