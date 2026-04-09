import json
import sys
import os

sys.path.append(os.path.abspath("."))

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


def build_title_to_docid(docs):
    return {doc.get("title", "").strip(): i for i, doc in enumerate(docs)}


def main():
    docs = load_corpus("data/processed_articles.json")
    queries = load_queries("queries.txt")

    bm25f = BM25FIndex(docs)
    hybrid = HybridSearchEngine(docs)

    title_to_docid = build_title_to_docid(docs)

    for qid, query in queries.items():
        print("\n" + "=" * 80)
        print(f"{qid}: {query}")
        print("=" * 80)

        print("\nBM25F top results:")
        _, bm25_results = bm25f.search(query, top_k=10)
        for r in bm25_results:
            print(
                f"doc_id={r.doc_id:<3} | rank={r.rank:<2} | score={r.score:.4f} | {r.title}"
            )

        print("\nHybrid top results:")
        _, hybrid_results = hybrid.search(
            query,
            candidate_count=100,
            bert_rerank_count=20,
            final_count=10
        )

        for i, r in enumerate(hybrid_results, start=1):
            doc_id = title_to_docid.get(r.title.strip(), "UNKNOWN")
            print(
                f"doc_id={doc_id:<3} | rank={i:<2} | final={r.final_score:.4f} | {r.title}"
            )


if __name__ == "__main__":
    main()
