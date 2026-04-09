"""Shared preprocessing utilities for the IR assignment pipeline."""

from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


REQUIRED_NLTK_PACKAGES = (
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
)


def ensure_nltk_resources(download_dir: str | None = None) -> None:
    import os
    from pathlib import Path
    
    # Use a default NLTK data directory if none specified
    if download_dir is None:
        download_dir = os.path.expanduser("~/nltk_data")
    
    # Ensure download directory exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    for resource_path, package_name in REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource '{package_name}'...")
            try:
                nltk.download(package_name, quiet=False, download_dir=download_dir)
            except Exception as download_error:
                print(f"Download error: {download_error}")
            
            # Verify the resource is now available
            try:
                nltk.data.find(resource_path)
            except LookupError as e:
                raise RuntimeError(
                    f"Failed to find NLTK resource '{package_name}' after download attempt. "
                    f"Download directory: {download_dir}. "
                    f"Please check your internet connection and try again."
                ) from e


def tokenize_text(text: str, stemmer: PorterStemmer, stopword_set: set[str]) -> tuple[list[str], list[str]]:
    raw_tokens = [token for token in word_tokenize(text) if re.search(r"\w", token)]
    lowered_tokens = [token.lower() for token in raw_tokens]
    filtered_tokens = []
    for token in lowered_tokens:
        normalized = token.strip("'")
        if not normalized:
            continue
        if normalized in {"s", "t", "re", "ve", "ll", "d", "m"}:
            continue
        filtered_tokens.append(normalized)
    processed_tokens = [
        stemmer.stem(token)
        for token in filtered_tokens
        if token not in stopword_set and re.search(r"[a-z0-9]", token)
    ]
    return filtered_tokens, processed_tokens


def preprocess_query(query: str) -> list[str]:
    stemmer = PorterStemmer()
    stopword_set = set(stopwords.words("english"))
    _, processed_tokens = tokenize_text(query, stemmer, stopword_set)
    return processed_tokens
