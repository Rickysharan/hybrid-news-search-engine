#!/usr/bin/env python3
"""Segment B: BM25F indexing and query pipeline."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from preprocessing import ensure_nltk_resources, preprocess_query


@dataclass
class SearchResult:
    doc_id: int
    rank: int
    score: float
    title: str
    source: str
    timestamp: str
    source_url: str


class BM25FIndex:
    def __init__(
        self,
        documents: list[dict],
        *,
        title_weight: float = 1.5,
        body_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.k1 = k1
        self.b = b
        self.num_documents = len(documents)
        self.title_lengths = [len(doc["processed_title_tokens"]) for doc in documents]
        self.body_lengths = [len(doc["processed_body_tokens"]) for doc in documents]
        self.avg_title_length = sum(self.title_lengths) / self.num_documents if documents else 0.0
        self.avg_body_length = sum(self.body_lengths) / self.num_documents if documents else 0.0
        self.title_term_freqs = [Counter(doc["processed_title_tokens"]) for doc in documents]
        self.body_term_freqs = [Counter(doc["processed_body_tokens"]) for doc in documents]
        self.title_index, self.body_index, self.document_frequencies = self._build_inverted_index()

    def _build_inverted_index(
        self,
    ) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, int]]:
        """Build per-field posting lists and document-frequency counts.

        title_index / body_index: term -> sorted list of doc_ids that contain it.
        document_frequencies:     term -> number of docs the term appears in.
        """
        title_index: dict[str, list[int]] = defaultdict(list)
        body_index: dict[str, list[int]] = defaultdict(list)
        document_frequencies: dict[str, int] = defaultdict(int)
        for doc_id, (title_freqs, body_freqs) in enumerate(
            zip(self.title_term_freqs, self.body_term_freqs)
        ):
            for term in title_freqs:
                title_index[term].append(doc_id)
            for term in body_freqs:
                body_index[term].append(doc_id)
            for term in set(title_freqs) | set(body_freqs):
                document_frequencies[term] += 1
        return dict(title_index), dict(body_index), dict(document_frequencies)

    def _field_length_norm(self, field_length: int, average_length: float) -> float:
        if average_length == 0:
            return 1.0
        return 1.0 - self.b + self.b * (field_length / average_length)

    def _weighted_tf(self, doc_id: int, term: str) -> float:
        title_tf = self.title_term_freqs[doc_id].get(term, 0)
        body_tf = self.body_term_freqs[doc_id].get(term, 0)
        title_norm = self._field_length_norm(self.title_lengths[doc_id], self.avg_title_length)
        body_norm = self._field_length_norm(self.body_lengths[doc_id], self.avg_body_length)
        weighted_title_tf = self.title_weight * (title_tf / title_norm if title_norm else 0.0)
        weighted_body_tf = self.body_weight * (body_tf / body_norm if body_norm else 0.0)
        return weighted_title_tf + weighted_body_tf

    def _idf(self, term: str) -> float:
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.num_documents - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 100) -> tuple[list[str], list[SearchResult]]:
        query_terms = preprocess_query(query)
        if not query_terms:
            return [], []

        # Use posting lists to restrict scoring to docs that contain at least one query term.
        candidate_doc_ids: set[int] = set()
        for term in query_terms:
            candidate_doc_ids.update(self.title_index.get(term, []))
            candidate_doc_ids.update(self.body_index.get(term, []))

        scores: list[tuple[int, float]] = []
        for doc_id in candidate_doc_ids:
            score = 0.0
            for term in query_terms:
                weighted_tf = self._weighted_tf(doc_id, term)
                if weighted_tf <= 0:
                    continue
                score += self._idf(term) * ((weighted_tf * (self.k1 + 1.0)) / (weighted_tf + self.k1))
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        results = [
            SearchResult(
                doc_id=doc_id,
                rank=rank,
                score=score,
                title=self.documents[doc_id]["title"],
                source=self.documents[doc_id]["source"],
                timestamp=self.documents[doc_id]["timestamp"],
                source_url=self.documents[doc_id]["source_url"],
            )
            for rank, (doc_id, score) in enumerate(scores[:top_k], start=1)
        ]
        return query_terms, results


def load_documents(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Processed corpus not found at {corpus_path}. Run segment_a_crawler.py first."
        )
    return json.loads(corpus_path.read_text(encoding="utf-8"))


def print_results(query: str, query_terms: list[str], results: list[SearchResult], display_count: int) -> None:
    print(f"Query: {query}")
    print(f"Processed query tokens: {query_terms}")
    print(f"Total hits: {len(results)}")
    for result in results[:display_count]:
        print(f"\nRank {result.rank} | Score {result.score:.4f}")
        print(f"  Title: {result.title}")
        print(f"  Source: {result.source}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"  URL: {result.source_url}")


SAMPLE_QUERIES = [
    "UK interest rate decision",
    "US election results",
    "climate change summit",
    "artificial intelligence technology",
    "stock market economy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the processed news corpus with BM25F.")
    parser.add_argument(
        "--corpus",
        default="data/processed_articles.json",
        help="Path to the preprocessed JSON corpus generated by Segment A.",
    )
    parser.add_argument(
        "--query",
        default="UK interest rate decision",
        help="Query to run against the BM25F index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Maximum number of ranked candidates to return.",
    )
    parser.add_argument(
        "--display-count",
        type=int,
        default=10,
        help="How many ranked results to print.",
    )
    parser.add_argument(
        "--title-weight",
        type=float,
        default=1.5,
        help="BM25F title field weight.",
    )
    parser.add_argument(
        "--body-weight",
        type=float,
        default=1.0,
        help="BM25F body field weight.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 term saturation parameter.",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 length normalization parameter.",
    )
    parser.add_argument(
        "--nltk-download-dir",
        default=None,
        help="Optional custom directory for NLTK resource downloads.",
    )
    parser.add_argument(
        "--run-sample-queries",
        action="store_true",
        help="Run all 5 sample queries and print results for manual verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_nltk_resources(download_dir=args.nltk_download_dir)
    documents = load_documents(Path(args.corpus))
    print(f"Loaded {len(documents)} documents from corpus.")
    print(f"BM25F params: k1={args.k1}, b={args.b}, title_weight={args.title_weight}, body_weight={args.body_weight}")
    index = BM25FIndex(
        documents,
        title_weight=args.title_weight,
        body_weight=args.body_weight,
        k1=args.k1,
        b=args.b,
    )
    print(f"Inverted index built: {len(index.title_index)} title terms, {len(index.body_index)} body terms.\n")

    queries_to_run = SAMPLE_QUERIES if args.run_sample_queries else [args.query]
    for query in queries_to_run:
        print("=" * 60)
        query_terms, results = index.search(query, top_k=args.top_k)
        print_results(query, query_terms, results, display_count=args.display_count)
        print()


if __name__ == "__main__":
    main()
