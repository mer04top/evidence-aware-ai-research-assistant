# Evidence-aware AI research assistant

A research assistant that checks its own answers against the scientific literature before showing them.

Pipeline: question → hybrid retrieval (BM25 + dense embeddings) → draft answer (LLM) → split into claims (LLM) → per-claim evidence retrieval → verification (NLI model, with an LLM fallback).

Status: work in progress. Retrieval, drafting, claim decomposition, corpus ingestion and verification run from the command line (`demo.py`). Aggregation, the API and the frontend are not finished.

## Setup

Python 3.12. A CPU is enough.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `LLM_API_KEY`. Any OpenAI-compatible API works; the example uses Groq. `HF_TOKEN` is optional and only avoids Hugging Face download warnings and rate limits.

## Build the corpus

The papers are not in the repository. Put PDF files in `data/papers/`, then run:

```
python -m app.corpus.build_corpus
```

This writes `data/corpus.json`. So far the corpus has seven papers: Lewis 2020 (RAG), FActScore 2023, VeriScore 2024, SciFact 2020, SciFact-Open 2022, Wallat 2025 and Wang 2025.

## Run the demo

```
python demo.py
```

[shows current working, for testing purposes only. will be replaced by the full working web application]\
It asks for a research question, then prints the retrieved passages, the drafted answer, the extracted claims and a verdict for each claim and passage. The first run downloads the embedding and NLI models. On a CPU, verification takes about 4 seconds per claim and passage pair, so an answer with many claims can take minutes.

## Layout

- `app/retrieval`: BM25, dense retrieval, fusion, `search()`
- `app/generation`: LLM client, prompts, answer drafting
- `app/decomposition`: splitting an answer into claims
- `app/corpus`: PDF ingestion, chunking, corpus building
- `app/verification`: NLI model, LLM fallback, `verify()`