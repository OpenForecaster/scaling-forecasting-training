from __future__ import annotations
"""Lightweight JSONL loader and field extractors for documents and questions.

This module provides:
- Robust parsing of various timestamp formats to epoch seconds.
- Projection helpers to shape raw JSON objects into the fields needed
  by the pipeline.
- A fast JSONL reader with optional per-record transform.
"""

try:
    import orjson as _fastjson
except Exception:
    _fastjson = None
import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from datetime import datetime, timezone


def _parse_epoch_seconds_data(value: Any) -> int | None:
    """Parse a date/time-like value into epoch seconds (UTC).

    Supports ISO-8601 strings (with optional timezone, space or 'T' separator),
    a trailing 'Z', and numeric Unix timestamps in seconds or milliseconds.
    Returns None for missing/empty values.
    """
    if value is None:
        return None

    # Numeric types: interpret as seconds unless clearly milliseconds
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1e12:  # very likely milliseconds
            val = val / 1000.0
        return int(val)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        # Normalize 'Z' UTC suffix if present
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        # Try Python's ISO parser which supports 'YYYY-MM-DD HH:MM:SS+HH:MM'
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # Try replacing space with 'T' if needed
            try:
                dt = datetime.fromisoformat(s.replace(" ", "T"))
            except ValueError:
                return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    
    return None

def _parse_epoch_seconds_questions(value: Any) -> int | None:
    """Parse question start dates into epoch seconds (UTC).

    - Returns None for missing or 'UNKNOWN'.
    - For date-only strings like 'YYYY-MM-DD', appends midnight UTC.
    - Otherwise supports the same formats as `_parse_epoch_seconds_data`.
    """
    if value is None or value == "UNKNOWN":
        return None

    # Numeric types: interpret as seconds unless clearly milliseconds
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1e12:  # very likely milliseconds
            val = val / 1000.0
        return int(val)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # If it's a bare date, add a midnight UTC time suffix
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            s = s + " 00:00:00+00:00"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = datetime.fromisoformat(s.replace(" ", "T"))
            except ValueError:
                return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    return None


def _extract_data_fields(obj: Dict[str, Any]) -> Tuple[str | None, Dict[str, Any]]:
    """Project document item to required fields with normalized `max_date`.

    Returns a `(key, value)` pair where `key` is the document id (or None if
    unavailable) and `value` contains the selected fields.
    """
    
    if obj.get("max_date", "NA") == "NA":
        url = obj.get("url")
        date_publish = obj.get("date_publish")
        date_modify = obj.get("date_modify")
        
        # Separately handle The Guardian's 2025 articles
        if "theguardian" in url and date_publish and "2025" in date_publish:
            pass 
        else :
            if obj.get("date_download"):
                date_publish = obj.get("date_download")
                
        if not date_publish and date_modify:
            obj["max_date"] = _parse_epoch_seconds_data(date_modify)
        elif date_publish and not date_modify:
            obj["max_date"] = _parse_epoch_seconds_data(date_publish)
        else:
            obj["max_date"] = max(_parse_epoch_seconds_data(date_publish), _parse_epoch_seconds_data(date_modify))
    
    return obj.get("id"), {
        "max_date": _parse_epoch_seconds_data(obj.get("max_date")),
        "authors": obj.get("authors"),
        "description": obj.get("description"),
        "maintext": obj.get("maintext"),
        "source_domain": obj.get("source_domain"),
        "title": obj.get("title"),
    }


def _extract_data_fields_questions(obj: Dict[str, Any]) -> Tuple[str | None, Dict[str, Any]]:
    """Project question item to required fields with normalized start date."""
    # if obj.get("max_date", "NA") == "NA":
    #     date_publish = obj.get("date_publish", obj.get("article_date_publish", obj.get("article_publish_date")))
    #     date_modify = obj.get("date_modify", obj.get("article_date_modify", obj.get("article_modify_date")))
    #     obj["max_date"] = max(_parse_epoch_seconds_data(date_publish), _parse_epoch_seconds_data(date_modify))
    

    ret = dict(obj)
    ret["question_start_date"] = _parse_epoch_seconds_questions(obj.get("question_start_date", obj.get("date_begin")))
    ret["resolution_date"] = _parse_epoch_seconds_questions(obj.get("resolution_date", obj.get("date_resolve_at")))
    ret["url"] = obj.get("url", obj.get("article_url", ""))
    return None, ret

    return None, {
        "question_start_date": _parse_epoch_seconds_questions(obj.get("question_start_date", obj.get("date_begin"))),
        "question_title": obj.get("question_title", obj.get("question", "")),
        "background": obj.get("background"),
        "resolution_criteria": obj.get("resolution_criteria"),
        "resolution_date": _parse_epoch_seconds_questions(obj.get("resolution_date", obj.get("date_resolve_at"))),
        "answer_type": obj.get("answer_type"),
        "answer": obj.get("answer"),
        "data_source": obj.get("data_source"),
        "news_source": obj.get("news_source"),
        "original_file": obj.get("original_file"),
        "url": obj.get("url", obj.get("article_url", "")),
        # "article_max_date": obj.get("max_date"),
        # "resolution_date_response": obj.get("resolution_date_response"),
    }


def load_jsonl(jsonl_path: Path, transform: Callable[[Any], Tuple[str | None, Any]] | None = None) -> Dict[str, Any]:
    """Load a JSONL file into a dictionary.

    - Skips empty and comment lines.
    - If `transform` is provided, it must return a `(key, value)` tuple for
      each JSON object; if `key` is None a 1-based line number is used instead.
    - Without a transform, uses `obj['id']` when present, else the line number
      string as the key, mapping to the raw object.
    """
    results: Dict[str, Any] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = (_fastjson.loads(s) if _fastjson is not None else json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON on line {i} of {jsonl_path}: {e}") from e

            if transform is not None:
                try:
                    key, value = transform(obj)
                    if key is None:
                        key = str(i)
                except Exception as e:
                    raise ValueError(f"Failed to transform JSON on line {i} of {jsonl_path}: {e}") from e
            else:
                if isinstance(obj, dict) and "id" in obj:
                    key = str(obj["id"])
                else:
                    key = str(i)
                value = obj

            results[key] = value

    return results
