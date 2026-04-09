import sys
import os
import json
from collections import defaultdict
from pathlib import Path

import pytrec_eval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_b_bm25f import BM25FIndex
from segment_c_hybrid import HybridSearchEngine


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(path):
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, query = line.strip().split("\t", 1)
                queries[qid] = query
    return queries


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, _, docid, rel = line.strip().split()
                qrels[qid][docid] = int(rel)
    return qrels


def run_bm25f(index, queries):
    run = {}
    for qid, query in queries.items():
        _, results = index.search(query, top_k=10)
        run[qid] = {str(r.doc_id): float(r.score) for r in results}
    return run


def run_hybrid(engine, queries, docs):
    run = {}

    title_to_docid = {}
    for i, doc in enumerate(docs):
        title_to_docid[doc.get("title", "").strip()] = str(i)

    for qid, query in queries.items():
        _, results = engine.search(
            query,
            candidate_count=100,
            bert_rerank_count=20,
            final_count=10,
        )

        doc_scores = {}
        for r in results:
            title = getattr(r, "title", "").strip()
            if title in title_to_docid:
                doc_id = title_to_docid[title]
                doc_scores[doc_id] = float(r.final_score)

        run[qid] = doc_scores

    return run


def evaluate(qrels, run):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recall_10", "P_10"})
    results = evaluator.evaluate(run)

    avg = {"MAP": 0.0, "Recall": 0.0, "P@10": 0.0}
    n = len(results)

    for q in results.values():
        avg["MAP"] += q["map"]
        avg["Recall"] += q["recall_10"]
        avg["P@10"] += q["P_10"]

    for k in avg:
        avg[k] /= n if n else 1

    return avg


def main():
    project_root = Path(__file__).resolve().parent.parent

    docs = load_corpus(project_root / "data" / "processed_articles.json")
    queries = load_queries(project_root / "queries.txt")
    qrels = load_qrels(Path(__file__).resolve().parent / "qrels.txt")

    bm25f = BM25FIndex(docs)
    hybrid = HybridSearchEngine(docs)

    bm25_res = run_bm25f(bm25f, queries)
    hybrid_res = run_hybrid(hybrid, queries, docs)

    print("\nBM25F:", evaluate(qrels, bm25_res))
    print("Hybrid:", evaluate(qrels, hybrid_res))


if __name__ == "__main__":
    main()
