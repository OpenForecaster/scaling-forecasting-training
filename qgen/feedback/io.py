"""Strict input loading and atomic stage checkpoints."""

import json
from datetime import datetime
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, row):
    with Path(path).open('a') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_articles(path):
    articles = read_jsonl(path)
    seen = set()
    for index, article in enumerate(articles):
        for field in ("title", "maintext", "url", "date_publish"):
            if not isinstance(article.get(field), str) or not article[field].strip():
                raise ValueError(f"Article {index}: missing or empty {field}")
        for field in ("date_publish", "date_modify", "date_download"):
            if article.get(field):
                datetime.fromisoformat(article[field].replace("Z", "+00:00"))
        if article["url"] in seen:
            raise ValueError(f"Article {index}: duplicate URL {article['url']}")
        seen.add(article["url"])
    if not articles:
        raise ValueError("No articles found")
    return articles
