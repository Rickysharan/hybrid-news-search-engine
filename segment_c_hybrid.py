#!/usr/bin/env python3
"""Segment C: hybrid retrieval with BERT re-ranking, temporal boost, and source authority."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from preprocessing import ensure_nltk_resources
from segment_b_bm25f import BM25FIndex, load_documents


DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_WEIGHTS = {
    "bm25f": 0.35,
    "bert": 0.25,
    "temporal": 0.15,
    "authority": 0.10,
}
DEFAULT_AUTHORITY = {
    "BBC": 1.0,
    "NYT": 1.0,
}


@dataclass
class HybridResult:
    rank: int
    title: str
    source: str
    timestamp: str
    source_url: str
    final_score: float
    bm25f_score: float
    bm25f_normalized: float
    bert_raw_score: float
    bert_score: float
    bert_normalized: float
    temporal_boost: float
    source_authority: float


class TransformerCrossEncoder:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        print("Importing torch and transformers...", flush=True)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading tokenizer for {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model weights for {model_name}...", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model ready on device: {self.device}", flush=True)

    def close(self) -> None:
        if hasattr(self, "model"):
            self.model.to("cpu")
        if hasattr(self, "torch") and self.device.type == "mps":
            self.torch.mps.empty_cache()

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start : start + batch_size]
                queries = [pair[0] for pair in batch_pairs]
                documents = [pair[1] for pair in batch_pairs]
                print(
                    f"Scoring batch {start + 1}-{start + len(batch_pairs)} of {len(pairs)}",
                    flush=True,
                )
                encoded = self.tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits.squeeze(dim=1)
                elif logits.ndim == 2:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits
                scores.extend(float(score) for score in batch_scores.detach().cpu())
        return scores


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def compute_age_in_days(timestamp: str) -> float:
    published_at = datetime.fromisoformat(timestamp)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    age_seconds = max(0.0, (now - published_at.astimezone(UTC)).total_seconds())
    return age_seconds / 86400.0


def build_document_text(document: dict) -> str:
    title = document.get("title", "").strip()
    body = document.get("body", "").strip()
    if body:
        return f"{title}. {body}"
    return title


class HybridSearchEngine:
    def __init__(
        self,
        documents: list[dict],
        *,
        title_weight: float = 1.5,
        body_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
        model_name: str = DEFAULT_MODEL_NAME,
        lambda_decay: float = 0.05,
        authority_default: float = 0.7,
        bm25f_weight: float = DEFAULT_WEIGHTS["bm25f"],
        bert_weight: float = DEFAULT_WEIGHTS["bert"],
        temporal_weight: float = DEFAULT_WEIGHTS["temporal"],
        authority_weight: float = DEFAULT_WEIGHTS["authority"],
    ) -> None:
        self.documents = documents
        self.bm25f_index = BM25FIndex(
            documents,
            title_weight=title_weight,
            body_weight=body_weight,
            k1=k1,
            b=b,
        )
        print(f"Loading cross-encoder model: {model_name}", flush=True)
        self.cross_encoder = TransformerCrossEncoder(model_name)
        self.lambda_decay = lambda_decay
        self.authority_default = authority_default
        self.bm25f_weight = bm25f_weight
        self.bert_weight = bert_weight
        self.temporal_weight = temporal_weight
        self.authority_weight = authority_weight

    def close(self) -> None:
        self.cross_encoder.close()

    def _source_authority(self, source: str) -> float:
        return DEFAULT_AUTHORITY.get(source, self.authority_default)

    def search(self, query: str, *, candidate_count: int = 100, bert_rerank_count: int = 20, final_count: int = 10) -> tuple[list[str], list[HybridResult]]:
        query_terms, candidates = self.bm25f_index.search(query, top_k=candidate_count)
        if not candidates:
            return query_terms, []

        print(f"Running BM25F candidate reranking for {len(candidates)} documents", flush=True)
        pairs = []
        for candidate in candidates:
            document = self.documents[candidate.doc_id]
            pairs.append((query, build_document_text(document)))

        bert_raw_scores = [float(score) for score in self.cross_encoder.predict(pairs)]
        bert_scores = [sigmoid(score) for score in bert_raw_scores]
        # Step 1: BERT re-ranks top 100 → top bert_rerank_count (default 20)
        bert_ranked = sorted(
            range(len(candidates)), key=lambda i: bert_scores[i], reverse=True
        )[:bert_rerank_count]
        print(f"BERT cut top {len(candidates)} → top {len(bert_ranked)} candidates", flush=True)

        # Step 2: normalise scores within the bert_rerank_count pool, combine all signals → top final_count
        pool_bert_scores = [bert_scores[i] for i in bert_ranked]
        pool_bm25f_scores = [candidates[i].score for i in bert_ranked]
        pool_bert_normalized = min_max_normalize(pool_bert_scores)
        pool_bm25f_normalized = min_max_normalize(pool_bm25f_scores)

        combined: list[HybridResult] = []
        for pool_pos, orig_idx in enumerate(bert_ranked):
            candidate = candidates[orig_idx]
            document = self.documents[candidate.doc_id]
            temporal_boost = math.exp(-self.lambda_decay * compute_age_in_days(document["timestamp"]))
            authority = self._source_authority(document["source"])
            final_score = (
                self.bm25f_weight * pool_bm25f_normalized[pool_pos]
                + self.bert_weight * pool_bert_normalized[pool_pos]
                + self.temporal_weight * temporal_boost
                + self.authority_weight * authority
            )
            combined.append(
                HybridResult(
                    rank=pool_pos + 1,
                    title=document["title"],
                    source=document["source"],
                    timestamp=document["timestamp"],
                    source_url=document["source_url"],
                    final_score=final_score,
                    bm25f_score=candidate.score,
                    bm25f_normalized=pool_bm25f_normalized[pool_pos],
                    bert_raw_score=bert_raw_scores[orig_idx],
                    bert_score=pool_bert_scores[pool_pos],
                    bert_normalized=pool_bert_normalized[pool_pos],
                    temporal_boost=temporal_boost,
                    source_authority=authority,
                )
            )

        combined.sort(key=lambda item: item.final_score, reverse=True)
        for index, result in enumerate(combined, start=1):
            result.rank = index
        return query_terms, combined[:final_count]


def print_results(query: str, query_terms: list[str], results: list[HybridResult]) -> None:
    print(f"Query: {query}")
    print(f"Processed query tokens: {query_terms}")
    print(f"Returned results: {len(results)}")
    for result in results:
        print(f"\nRank {result.rank} | Final score {result.final_score:.4f}")
        print(f"  Title: {result.title}")
        print(f"  Source: {result.source}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"  URL: {result.source_url}")
        print(
            "  Score breakdown: "
            f"BM25F={result.bm25f_score:.4f} (norm {result.bm25f_normalized:.4f}), "
            f"BERT={result.bert_score:.4f} raw={result.bert_raw_score:.4f} "
            f"(norm {result.bert_normalized:.4f}), "
            f"temporal={result.temporal_boost:.4f}, authority={result.source_authority:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval with BM25F + BERT + temporal/source boosts.")
    parser.add_argument(
        "--corpus",
        default="data/processed_articles.json",
        help="Path to the preprocessed JSON corpus generated by Segment A.",
    )
    parser.add_argument(
        "--query",
        default="UK interest rate decision",
        help="Query to run against the hybrid search engine.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=100,
        help="How many BM25F candidates to pass into BERT re-ranking.",
    )
    parser.add_argument(
        "--final-count",
        type=int,
        default=10,
        help="How many final ranked results to print.",
    )
    parser.add_argument(
        "--bert-rerank-count",
        type=int,
        default=20,
        help="How many top BERT-scored candidates to keep before combining all signals.",
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
        "--lambda-decay",
        type=float,
        default=0.05,
        help="Exponential temporal decay rate in days.",
    )
    parser.add_argument(
        "--authority-default",
        type=float,
        default=0.7,
        help="Fallback authority score for unknown sources.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face cross-encoder model name.",
    )
    parser.add_argument(
        "--nltk-download-dir",
        default=None,
        help="Optional custom directory for NLTK resource downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Preparing NLTK resources...", flush=True)
    ensure_nltk_resources(download_dir=args.nltk_download_dir)
    print(f"Loading corpus from {args.corpus}...", flush=True)
    documents = load_documents(Path(args.corpus))
    print(f"Loaded {len(documents)} documents", flush=True)
    engine = HybridSearchEngine(
        documents,
        title_weight=args.title_weight,
        body_weight=args.body_weight,
        k1=args.k1,
        b=args.b,
        model_name=args.model_name,
        lambda_decay=args.lambda_decay,
        authority_default=args.authority_default,
    )
    query_terms, results = engine.search(
        args.query,
        candidate_count=args.candidate_count,
        bert_rerank_count=args.bert_rerank_count,
        final_count=args.final_count,
    )
    print_results(args.query, query_terms, results)


if __name__ == "__main__":
    main()
