# IR Assignment - Segment A

This repository currently contains `Segment A`, `Segment B`, and `Segment C` from the Assignment 2 blueprint:

- an RSS crawler plus an NLTK preprocessing pipeline
- a BM25F-style index and query pipeline over the processed corpus
- a hybrid ranker with BERT re-ranking, temporal boost, and source authority

## Files

- `segment_a_crawler.py` - fetches RSS articles, preprocesses title and body text, and writes JSON output
- `segment_b_bm25f.py` - loads the processed corpus and returns ranked BM25F search results
- `segment_c_hybrid.py` - reranks BM25F candidates with a cross-encoder and scoring boosts
- `preprocessing.py` - shared NLTK resource and tokenization helpers
- `requirements.txt` - Python dependencies for this segment

## What the script does

1. Polls the BBC and NYT RSS feeds.
2. Extracts each article as a structured record with:
   - `title`
   - `body`
   - `timestamp`
   - `source`
   - `source_url`
3. Applies the agreed NLTK pipeline:
   - tokenisation
   - lowercasing
   - stopword removal
   - Porter stemming
4. Stores one processed record per article in JSON.
5. Prints 5 sample articles so the output can be checked manually.

## Install

```bash
python3 -m pip install -r requirements.txt
```

If `torch` or `transformers` misbehave under Homebrew Python 3.14, use Python 3.13 for Segment C:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv313
.venv313/bin/python -m pip install -r requirements.txt
```

## Run

```bash
python3 segment_a_crawler.py
```

Optional flags:

```bash
python3 segment_a_crawler.py --per-feed-limit 15 --sample-count 5 --output data/processed_articles.json
```

## Output

The script writes the processed corpus to:

```text
data/processed_articles.json
```

Each JSON record includes both raw tokens and processed tokens for `title` and `body`, which keeps Segment B straightforward when building the BM25F index.

## Segment B

Run BM25F retrieval against the processed corpus:

```bash
python3 segment_b_bm25f.py --query "UK interest rate decision" --top-k 100 --display-count 10
```

The retrieval module:

1. Loads the preprocessed corpus from Segment A.
2. Builds per-field term frequencies for title and body.
3. Applies the same NLTK preprocessing pipeline to the user query.
4. Scores documents with BM25F-style weighted term frequencies using:
   - `w_title = 1.5`
   - `w_body = 1.0`
   - `k1 = 1.5`
   - `b = 0.75`
5. Returns up to the top 100 candidates, ready for later BERT re-ranking.

## Segment C

Run hybrid retrieval against the processed corpus:

```bash
python3 segment_c_hybrid.py --query "UK interest rate decision" --candidate-count 100 --final-count 10
```

The hybrid module:

1. Takes the top BM25F candidates from Segment B.
2. Scores each `(query, document)` pair with `cross-encoder/ms-marco-MiniLM-L-6-v2` through Hugging Face `transformers`.
3. Applies temporal freshness with `exp(-lambda * age_in_days)`, using `lambda = 0.05` by default.
4. Applies source authority with `BBC = 1.0`, `NYT = 1.0`, `unknown = 0.7`.
5. Combines the signals with the agreed weights:
   - `0.35 * BM25F`
   - `0.25 * BERT`
   - `0.15 * temporal`
   - `0.10 * authority`
6. Prints the final top 10 results with a visible score breakdown for demo use.

## Query GUI

Launch a simple desktop GUI to test queries interactively:

```bash
python3 query_gui.py
```

Options:

```bash
python3 query_gui.py --mode bm25f --results 10
python3 query_gui.py --mode hybrid --results 10
```

Inside the GUI, you can type custom queries such as:

- `UK interest rate decision`
- `Gaza ceasefire talks`
- `Keir Starmer immigration policy`

Notes:

- `hybrid` mode loads the BERT cross-encoder on first use, so the first query can take longer.
- `bm25f` mode is much faster for quick interactive testing.
