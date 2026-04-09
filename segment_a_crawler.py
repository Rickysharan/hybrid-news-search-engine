15#!/usr/bin/env python3
"""Segment A: RSS crawler and NLTK preprocessing pipeline.

Builds a small news corpus from BBC and NYT RSS feeds, applies the shared
preprocessing pipeline, and writes the output as JSON for later indexing.
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

import certifi
import feedparser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from preprocessing import ensure_nltk_resources, tokenize_text


DEFAULT_FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
    "NYT": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
}


@dataclass
class RawArticle:
    title: str
    body: str
    timestamp: str
    source: str
    source_url: str


@dataclass
class ProcessedArticle:
    title: str
    body: str
    timestamp: str
    source: str
    source_url: str
    raw_title_tokens: list[str]
    raw_body_tokens: list[str]
    processed_title_tokens: list[str]
    processed_body_tokens: list[str]


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def parse_timestamp(entry: dict[str, Any]) -> str:
    candidates = [
        as_text(entry.get("published")),
        as_text(entry.get("updated")),
        as_text(entry.get("pubDate")),
        as_text(entry.get("created")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            dt = parsedate_to_datetime(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC).isoformat()
        except (TypeError, ValueError, IndexError):
            continue
    return datetime.now(UTC).isoformat()


def extract_body(entry: dict[str, Any]) -> str:
    summary = as_text(entry.get("summary"))
    if summary:
        return clean_html(summary)
    description = as_text(entry.get("description"))
    if description:
        return clean_html(description)
    content = entry.get("content") or []
    if content:
        first_content = content[0]
        if isinstance(first_content, dict):
            return clean_html(as_text(first_content.get("value")))
    return ""


def fetch_feed(url: str) -> bytes:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    request = Request(url, headers={"User-Agent": "IR-Assignment-NewsCrawler/1.0"})
    with urlopen(request, context=ssl_context, timeout=30) as response:
        return response.read()


def fetch_articles(feed_map: dict[str, str], per_feed_limit: int) -> list[RawArticle]:
    articles: list[RawArticle] = []
    for source, url in feed_map.items():
        try:
            parsed = feedparser.parse(fetch_feed(url))
        except Exception as exc:
            print(f"Skipping {source} feed due to parsing error: {exc}")
            continue

        if not getattr(parsed, "entries", None):
            print(f"Skipping {source} feed because no entries were found")
            continue

        for entry in parsed.entries[:per_feed_limit]:
            articles.append(
                RawArticle(
                    title=clean_html(as_text(entry.get("title"))),
                    body=extract_body(entry),
                    timestamp=parse_timestamp(entry),
                    source=source,
                    source_url=as_text(entry.get("link")).strip(),
                )
            )
    return articles


def preprocess_articles(raw_articles: Iterable[RawArticle]) -> list[ProcessedArticle]:
    stemmer = PorterStemmer()
    stopword_set = set(stopwords.words("english"))
    processed_articles: list[ProcessedArticle] = []
    for article in raw_articles:
        raw_title_tokens, processed_title_tokens = tokenize_text(article.title, stemmer, stopword_set)
        raw_body_tokens, processed_body_tokens = tokenize_text(article.body, stemmer, stopword_set)
        processed_articles.append(
            ProcessedArticle(
                title=article.title,
                body=article.body,
                timestamp=article.timestamp,
                source=article.source,
                source_url=article.source_url,
                raw_title_tokens=raw_title_tokens,
                raw_body_tokens=raw_body_tokens,
                processed_title_tokens=processed_title_tokens,
                processed_body_tokens=processed_body_tokens,
            )
        )
    return processed_articles


def write_corpus(articles: Iterable[ProcessedArticle], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(article) for article in articles]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_samples(articles: list[ProcessedArticle], sample_count: int) -> None:
    for index, article in enumerate(articles[:sample_count], start=1):
        print(f"\nArticle {index}")
        print(f"  Source: {article.source}")
        print(f"  Title: {article.title}")
        print(f"  Timestamp: {article.timestamp}")
        print(f"  URL: {article.source_url}")
        print(f"  Processed title tokens: {article.processed_title_tokens}")
        print(f"  Processed body tokens: {article.processed_body_tokens[:30]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Segment A preprocessed news corpus.")
    parser.add_argument(
        "--output",
        default="data/processed_articles.json",
        help="Path to write the processed JSON corpus.",
    )
    parser.add_argument(
        "--per-feed-limit",
        type=int,
        default=15,
        help="Maximum number of articles to pull from each feed.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=30,#change to 30 to print more samples for verification
        help="Number of processed articles to print for manual verification.",
    )
    parser.add_argument(
        "--nltk-download-dir",
        default=None,
        help="Optional custom directory for NLTK resource downloads.",
    )
    return parser.parse_args()


def prompt_int(message: str, default: int) -> int:
    """Prompt the user for an integer, falling back to *default* on empty input."""
    try:
        raw = input(f"{message} [default: {default}]: ").strip()
    except EOFError:
        return default
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 1:
            print(f"Value must be at least 1, using default ({default}).")
            return default
        return value
    except ValueError:
        print(f"Invalid number entered, using default ({default}).")
        return default


def main() -> None:
    args = parse_args()

    if sys.stdin.isatty():
        # Keep the interactive prompts for direct terminal use, but avoid
        # breaking CLI and non-interactive runs where flags should be enough.
        per_feed_limit = prompt_int(
            "How many articles to fetch per feed (BBC + NYT)?", args.per_feed_limit
        )
        sample_count = prompt_int(
            "How many articles to print for manual verification?", args.sample_count
        )
    else:
        per_feed_limit = args.per_feed_limit
        sample_count = args.sample_count

    ensure_nltk_resources(download_dir=args.nltk_download_dir)
    raw_articles = fetch_articles(DEFAULT_FEEDS, per_feed_limit=per_feed_limit)
    processed_articles = preprocess_articles(raw_articles)
    write_corpus(processed_articles, Path(args.output))
    print(f"Wrote {len(processed_articles)} processed articles to {args.output}")
    print_samples(processed_articles, sample_count=sample_count)


if __name__ == "__main__":
    main()
