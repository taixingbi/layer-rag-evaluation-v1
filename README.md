# Layer RAG evaluation

Build gold JSONL from ingested Qdrant points and batch-score the live RAG gateway (`POST /v1/rag/query`): retrieval rank, `must_contain`, citation match, heuristic answer proxies, and latency.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env   # set RAG_BASE_URL and RAG_COLLECTION_BASE
```

`.env` is loaded on `import app.core.config`. **`RAG_BASE_URL` and `RAG_COLLECTION_BASE` are required** for `run_eval` (no hardcoded gateway defaults).

## Layout

```
app/
  core/           config.py, paths.py
  eval/           gold_dataset.py, run_eval.py, scoring.py, baseline.py
  http/           rag.py, inference.py
data_<env>/
  gold_dataset/   generated + hand-authored JSONL
  report/         eval outputs (gitignored; use --summary-json / --report-json)
docs/eval.md      workflow + CLI reference
tests/            unit tests (no live RAG)
.github/workflows/ci.yml
```

## What gets evaluated

| Layer | Module | Signals |
|-------|--------|---------|
| **Retrieval** | `app.eval.run_eval` | MRR, Recall@k, Precision@k, NDCG@k, F1@k from `retrieval_hits` |
| **Answer** | `app.eval.run_eval` | `must_contain`, citation `source`, **`heuristic_quality_*`** (proxy, not LLM judge) |
| **Latency** | `app.eval.run_eval` | p50 / p95 / p99 |

## Quick start (dev)

```bash
# 1. Gold JSONL (defaults to ../layer-rag-ingest-v1/data_dev only)
python -m app.eval.gold_dataset \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset

# 2. Eval + reports
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --summary-json data_dev/report/rag_eval_summary.json \
  --report-json data_dev/report/rag_eval_report.json

# 3. Optional regression gate vs pinned baseline
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --limit 50 \
  --baseline-json tests/fixtures/baseline_summary.json \
  --baseline-tolerance 0.05
```

Full workflow: [docs/eval.md](docs/eval.md).

## Configuration (`.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `RAG_BASE_URL` | **Yes** (for eval) | RAG gateway (no `/v1` suffix) |
| `RAG_COLLECTION_BASE` | **Yes** (for eval) | `collection_base` in API body |
| `CHAT_BASE_URL` / `INFERENCE_URL` | Only with `--enable-must-contain-llm` | Gold generator LLM extraction |

## Development

```bash
pytest
```

CI runs the same on push/PR to `main`.

## Related repos

- [layer-rag-ingest-v1](../layer-rag-ingest-v1) — ingest, synthetic questions, Qdrant upsert
- [layer-rag-query-v1](../layer-rag-query-v1) — RAG HTTP API contract
