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
  eval/           gold_dataset.py, run_eval.py, scoring.py, baseline.py, metadata.py, llm_judge.py
  http/           rag.py, inference.py
data_<env>/
  gold_dataset/   generated + hand-authored JSONL (gitignored; regenerate locally)
  report/         eval outputs (gitignored; use --summary-json / --report-json)
docs/eval.md      workflow + CLI reference
tests/            unit tests + fixtures (no live RAG)
.github/workflows/ci.yml
.github/workflows/rag-eval.yml   live eval (uses repository secrets)
```

## What gets evaluated

| Layer | Module | Signals |
|-------|--------|---------|
| **Retrieval** | `app.eval.run_eval` | MRR, Recall@k, Precision@k, NDCG@k, F1@k from `retrieval_hits` |
| **Answer (heuristic)** | `app.eval.run_eval` | `must_contain`, citation `source`, `heuristic_quality_*` (fast proxy) |
| **Answer (LLM judge)** | `app.eval.run_eval --enable-llm-judge` | `llm_judge_*` semantic scores (same run) |
| **Latency** | `app.eval.run_eval` | p50 / p95 / p99 |

## Quick start (dev)

```bash
# 1. Gold JSONL (defaults to ../layer-rag-ingest-v1/data_dev only)
python -m app.eval.gold_dataset \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset

# 2. Eval + reports (heuristic only)
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --summary-json data_dev/report/rag_eval_summary.json \
  --report-json data_dev/report/rag_eval_report.json

# 2b. Same run with LLM-as-judge (requires LLM_JUDGE_URL or INFERENCE_URL in .env)
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --enable-llm-judge \
  --llm-judge-concurrency 10 \
  --limit 20 \
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
| `LLM_JUDGE_URL` / `INFERENCE_URL` / `CHAT_BASE_URL` | With `--enable-llm-judge` | Chat API for semantic judge |
| `LLM_JUDGE_MODEL` / `CHAT_MODEL` | Optional | Judge model (default Qwen2.5-7B-Instruct) |
| `CHAT_BASE_URL` / `INFERENCE_URL` | With `--enable-must-contain-llm` | Gold generator LLM extraction |

## Development

```bash
pytest
ruff check app tests
```

CI runs the same on push/PR to `main`.

## GitHub Actions secrets

For the **RAG eval** workflow (`.github/workflows/rag-eval.yml`), add these **repository secrets**:

| Secret | Example (dev) | Purpose |
|--------|---------------|---------|
| `RAG_BASE_URL` | `http://192.168.86.179:30183` | RAG gateway base (no `/v1`) |
| `RAG_COLLECTION_BASE` | `taixing_knowledge` | `collection_base` in API body |

The runner must reach `RAG_BASE_URL` (private LAN requires a self-hosted runner or VPN-exposed URL).

Manual run: **Actions → RAG eval → Run workflow**. Default gold is `tests/fixtures/gold_single_row.jsonl` (smoke). For full regression, run locally with `data_dev/gold_dataset/` and `--baseline-json tests/fixtures/baseline_summary.json`.

## Related repos

- [layer-rag-ingest-v1](../layer-rag-ingest-v1) — ingest, synthetic questions, Qdrant upsert
- [layer-rag-query-v1](../layer-rag-query-v1) — RAG HTTP API contract
