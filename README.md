# Layer RAG evaluation

Build gold JSONL from ingested Qdrant points and batch-score the live RAG gateway (`POST /v1/rag/query`): retrieval rank, `must_contain`, citation match, answer-quality heuristics, and latency.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cp .env.example .env   # optional; edit RAG_BASE_URL, etc.
```

`.env` at the repo root is loaded on `import app` (via `app.core.config`).

## Repository layout

```
app/
  core/           config.py, paths.py (REPO_ROOT, INGEST_ROOT)
  eval/           gold_dataset.py, run_eval.py
  http/           rag.py (shared RAG client), inference.py (chat API for gold generator)
data_<env>/
  gold_dataset/   easy_single_hop.jsonl, paraphrase.jsonl, hand-authored splits
  report/         rag_eval_summary.json, rag_eval_report.json
docs/
  eval.md          workflow + CLI reference
```

Points input lives in sibling **[layer-rag-ingest-v1](../layer-rag-ingest-v1)** (`data_<env>/data1/processed/points_*.json`). Gold JSONL and reports stay in this repo under `data_<env>/`.

## What gets evaluated

| Layer | Module | Signals |
|-------|--------|---------|
| **Retrieval** | `app.eval.run_eval` | Rank, RR, MRR, Recall@k / Precision@k / NDCG@k / F1@k from `retrieval_hits` |
| **Answer** | `app.eval.run_eval` | `must_contain`, citation `source` match, heuristic `quality_dimensions` |
| **Latency** | `app.eval.run_eval` | `latency_ms` per row; p50 / p95 / p99 in summary |

## Quick start (dev)

Run from this repo root after ingest + Qdrant upsert (see [docs/eval.md](docs/eval.md)).

**1. Generate gold JSONL**

```bash
python -m app.eval.gold_dataset \
  --data-roots ../layer-rag-ingest-v1/data_dev \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset
```

**2. Smoke eval (10 rows)**

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --limit 10
```

**3. Full eval + reports**

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/ \
  --report-json data_dev/report/rag_eval_report.json \
  --summary-json data_dev/report/rag_eval_summary.json
```

By default, `--data-roots` scans `../layer-rag-ingest-v1/data_dev`, `data_qa`, and `data_prod`. Pass explicit roots (e.g. only `data_dev`) to avoid mixing envs or warnings for missing folders.

## Commands

| Command | Purpose |
|---------|---------|
| `python -m app.eval.gold_dataset` | Build / refresh gold JSONL from `points_*.json` |
| `python -m app.eval.run_eval` | Call RAG per gold row and aggregate scores |

Full documentation: [docs/eval.md](docs/eval.md).

## Configuration (`.env`)

| Variable | Used by |
|----------|---------|
| `RAG_BASE_URL` | Default `--rag-base-url` for `run_eval` (e.g. `http://192.168.86.179:30183`) |
| `RAG_COLLECTION_BASE` | Default `--collection-base` (e.g. `taixing_knowledge`) |
| `CHAT_BASE_URL` / `INFERENCE_URL` | `gold_dataset --enable-must-contain-llm` |
| `CHAT_MODEL` / `INFERENCE_MODEL` | Chat model for must_contain extraction |
| `CHAT_API_KEY` | Optional bearer token |

Upsert and Qdrant settings (`QDRANT_URL`, `COLLECTION_NAME`, `ENV`) belong in **layer-rag-ingest-v1** (e.g. `.env.dev`).

## Related repos

- [layer-rag-ingest-v1](../layer-rag-ingest-v1) — chunk, embed, synthetic questions, Qdrant upsert
- [layer-rag-query-v1](../layer-rag-query-v1) — RAG HTTP API, `retrieval_hits` contract ([eval.md](../layer-rag-query-v1/docs/eval.md))
