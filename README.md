# Layer RAG evaluation

Batch-evaluate a gold Q&A JSON against a local RAG server: fill `inference-output` via `POST /v1/rag/query`, then attach per-row metrics (LLM-as-judge when configured, or heuristics only).

## Prerequisites

- Python **3.11**
- A running RAG API exposing **`/v1/rag/query`** (see `tmp.md` for request/response shape)
- Optional: an OpenAI-compatible **`/v1/chat/completions`** endpoint for the metric judge (set in `.env`)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy or create a `.env` in the repo root if you use the LLM judge (see **Configuration**).

## Run

**End-to-end** (RAG fill + metrics in one step; writes RAG results then merges metrics into the same `-o` file):

```bash
python3 main.py -i dataset/dataset-gold-test-1.0.0.json -o result/dataset-gold-test-1.0.0.json --rag-max-concurrency 1 --judge-max-concurrency 32
```

Omit `-i` / `-o` to use the defaults under `dataset/` and `result/`.

### Useful flags

| Flag | Meaning |
|------|--------|
| `--base-url` | RAG server (default: `http://127.0.0.1:8000`) |
| `-c` / `--collection` | `collection_base` sent to RAG (default: `taixing_knowledge`) |
| `-k`, `--k-max` | Retrieval params (defaults: `5`, `40`) |
| `--timeout` | Per-request HTTP timeout in seconds (default: `300`) |
| `--rag-max-concurrency` | Max in-flight `/v1/rag/query` requests from this client (default: `20`) |
| `--judge-max-concurrency` | Max concurrent LLM judge requests (default: same as `--rag-max-concurrency`) |
| `--max-attempts`, `--retry-backoff` | Retries on transient RAG errors |
| `--heuristic-only` | Metrics step skips the LLM judge (string/embed heuristics only) |
| `--judge-max-attempts`, `--judge-retry-backoff` | Judge HTTP retries (defaults also from env) |

`--rag-max-concurrency` only limits how many concurrent `/v1/rag/query` requests the RAG step sends; total time still depends on the server. Progress lines print as each slot starts work. (`rag_query.py` / `metric.py` still use `--max-concurrency` for their respective HTTP clients.)

**Stop the run:** use **Ctrl+C**. Avoid **Ctrl+Z** (suspends the job; you may see `zsh: suspended`).

### Standalone steps

- RAG only: `python3 rag_query.py -h`
- Metrics only (e.g. on an existing result file): `python3 metric.py -h`

## Configuration (`.env`)

Loaded automatically for the **metric** step. If `LLM_JUDGE_URL` and `INFERENCE_URL` are both unset or empty, metrics use heuristics only. Use `--heuristic-only` to skip the judge even when a judge URL is configured.

| Variable | Purpose |
|----------|--------|
| `LLM_JUDGE_URL` or `INFERENCE_URL` | Base URL for OpenAI-compatible chat completions |
| `LLM_JUDGE_MODEL` or `INFERENCE_MODEL` | Model name for the judge |
| `LLM_JUDGE_TIMEOUT` | Request timeout (default: `120`) |
| `LLM_JUDGE_MAX_TOKENS` | `max_tokens` (default: `400`) |
| `LLM_JUDGE_MAX_ATTEMPTS` | Judge retries (default: `3`) |
| `LLM_JUDGE_RETRY_BACKOFF` | Base backoff seconds for judge (default: `1.0`) |

## Dataset format

Gold file: JSON array of objects with at least:

- `input` — question
- `output` — reference answer
- `inference-output` — filled by RAG (often empty in the gold file)

After `main.py`, each row includes RAG fields (e.g. answer, citations) and a `metrics` object when the metric step succeeds.
