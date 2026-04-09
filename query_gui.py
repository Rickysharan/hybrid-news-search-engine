#!/usr/bin/env python3
"""Simple GUI for running BM25F or hybrid queries over the news corpus."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError as exc:
    if exc.name == "_tkinter":
        raise SystemExit(
            "Tkinter is not available in this Python build.\n"
            "Run the GUI with a Tk-enabled interpreter instead, for example:\n"
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 query_gui.py"
        ) from exc
    raise

from preprocessing import ensure_nltk_resources
from segment_b_bm25f import BM25FIndex, SearchResult, load_documents
from segment_c_hybrid import DEFAULT_MODEL_NAME, HybridResult, HybridSearchEngine


SAMPLE_QUERIES = [
    "UK interest rate decision",
    "Gaza ceasefire talks",
    "Keir Starmer immigration policy",
]


class QueryGui(tk.Tk):
    def __init__(
        self,
        documents: list[dict],
        *,
        default_mode: str,
        default_results: int,
        model_name: str,
    ) -> None:
        super().__init__()
        self.title("News Query Tester")
        self.geometry("980x680")

        self.documents = documents
        self.model_name = model_name
        self.bm25_index = BM25FIndex(documents)
        self.hybrid_engine: HybridSearchEngine | None = None
        self.search_in_progress = False

        self.query_var = tk.StringVar(value=SAMPLE_QUERIES[0])
        self.mode_var = tk.StringVar(value=default_mode)
        self.results_var = tk.IntVar(value=default_results)
        self.status_var = tk.StringVar(value=f"Ready. Loaded {len(documents)} articles.")

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        ttk.Label(container, text="Enter query").grid(row=0, column=0, sticky="w")

        query_row = ttk.Frame(container)
        query_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        query_row.columnconfigure(0, weight=1)
        self.query_entry = ttk.Entry(query_row, textvariable=self.query_var)
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.query_entry.bind("<Return>", self._on_search)
        self.search_button = ttk.Button(query_row, text="Search", command=self._on_search)
        self.search_button.grid(row=0, column=1)

        options_row = ttk.Frame(container)
        options_row.grid(row=2, column=0, sticky="w", pady=(0, 8))

        ttk.Label(options_row, text="Mode:").grid(row=0, column=0, sticky="w")
        mode_box = ttk.Combobox(
            options_row,
            textvariable=self.mode_var,
            values=["bm25f", "hybrid"],
            state="readonly",
            width=10,
        )
        mode_box.grid(row=0, column=1, sticky="w", padx=(6, 16))

        ttk.Label(options_row, text="Top results:").grid(row=0, column=2, sticky="w")
        result_spin = ttk.Spinbox(
            options_row,
            from_=1,
            to=50,
            textvariable=self.results_var,
            width=6,
        )
        result_spin.grid(row=0, column=3, sticky="w", padx=(6, 0))

        samples_row = ttk.Frame(container)
        samples_row.grid(row=3, column=0, sticky="w", pady=(0, 8))
        ttk.Label(samples_row, text="Samples:").pack(side=tk.LEFT)
        for sample in SAMPLE_QUERIES:
            ttk.Button(
                samples_row,
                text=sample,
                command=lambda q=sample: self.query_var.set(q),
            ).pack(side=tk.LEFT, padx=(8, 0))

        result_frame = ttk.Frame(container)
        result_frame.grid(row=4, column=0, sticky="nsew")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = tk.Text(result_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky="nsew")
        result_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=result_scroll.set)

        ttk.Label(container, textvariable=self.status_var).grid(row=5, column=0, sticky="w", pady=(8, 0))

    def _on_search(self, _event: object | None = None) -> None:
        if self.search_in_progress:
            return

        query = self.query_var.get().strip()
        if not query:
            self.status_var.set("Please enter a query.")
            return

        mode = self.mode_var.get().strip().lower()
        try:
            top_k = max(1, min(50, int(self.results_var.get())))
        except (TypeError, ValueError):
            top_k = 10
            self.results_var.set(top_k)

        self.search_in_progress = True
        self.search_button.configure(state=tk.DISABLED)
        self.status_var.set(f"Searching with {mode.upper()}...")

        worker = threading.Thread(
            target=self._run_search,
            args=(query, mode, top_k),
            daemon=True,
        )
        worker.start()

    def _run_search(self, query: str, mode: str, top_k: int) -> None:
        try:
            if mode == "hybrid":
                if self.hybrid_engine is None:
                    self._set_status_main_thread("Loading hybrid model (first run may take a while)...")
                    self.hybrid_engine = HybridSearchEngine(self.documents, model_name=self.model_name)
                query_terms, results = self.hybrid_engine.search(
                    query,
                    final_count=top_k,
                )
                output = self._format_hybrid_results(query, query_terms, results)
            else:
                query_terms, results = self.bm25_index.search(query, top_k=top_k)
                output = self._format_bm25_results(query, query_terms, results)

            self.after(0, self._apply_results, output, f"Done. Returned {len(results)} results.")
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._apply_results, f"Error: {exc}", "Search failed.")

    def _set_status_main_thread(self, status_text: str) -> None:
        self.after(0, lambda: self.status_var.set(status_text))

    def _on_close(self) -> None:
        if self.hybrid_engine is not None:
            self.hybrid_engine.close()
            self.hybrid_engine = None
        self.destroy()

    def _apply_results(self, text: str, status: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)
        self.search_button.configure(state=tk.NORMAL)
        self.search_in_progress = False
        self.status_var.set(status)

    @staticmethod
    def _format_bm25_results(query: str, query_terms: list[str], results: list[SearchResult]) -> str:
        lines = [
            f"Query: {query}",
            f"Processed query tokens: {query_terms}",
            f"Total hits: {len(results)}",
            "",
        ]
        for result in results:
            lines.append(f"Rank {result.rank} | Score {result.score:.4f}")
            lines.append(f"Title: {result.title}")
            lines.append(f"Source: {result.source}")
            lines.append(f"Timestamp: {result.timestamp}")
            lines.append(f"URL: {result.source_url}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_hybrid_results(query: str, query_terms: list[str], results: list[HybridResult]) -> str:
        lines = [
            f"Query: {query}",
            f"Processed query tokens: {query_terms}",
            f"Returned results: {len(results)}",
            "",
        ]
        for result in results:
            lines.append(f"Rank {result.rank} | Final score {result.final_score:.4f}")
            lines.append(f"Title: {result.title}")
            lines.append(f"Source: {result.source}")
            lines.append(f"Timestamp: {result.timestamp}")
            lines.append(f"URL: {result.source_url}")
            lines.append(
                "Score breakdown: "
                f"BM25F={result.bm25f_score:.4f} (norm {result.bm25f_normalized:.4f}), "
                f"BERT={result.bert_score:.4f} raw={result.bert_raw_score:.4f} "
                f"(norm {result.bert_normalized:.4f}), "
                f"temporal={result.temporal_boost:.4f}, authority={result.source_authority:.4f}"
            )
            lines.append("")
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a simple GUI for query testing.")
    parser.add_argument(
        "--corpus",
        default="data/processed_articles.json",
        help="Path to the preprocessed JSON corpus generated by Segment A.",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25f", "hybrid"],
        default="bm25f",
        help="Default search mode selected in the GUI.",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=10,
        help="Default number of top results to return.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face cross-encoder model name used for hybrid mode.",
    )
    parser.add_argument(
        "--nltk-download-dir",
        default=None,
        help="Optional custom directory for NLTK resource downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_nltk_resources(download_dir=args.nltk_download_dir)
    documents = load_documents(Path(args.corpus))
    app = QueryGui(
        documents,
        default_mode=args.mode,
        default_results=max(1, min(50, args.results)),
        model_name=args.model_name,
    )
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app._on_close()


if __name__ == "__main__":
    main()
