# Layer RAG evaluation

Build gold JSONL from ingested Qdrant points and batch-score the live RAG gateway (`POST /v1/rag/query`): retrieval rank, `must_contain`, citation match, heuristic answer proxies, and latency.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env   # RAG_* required; Supabase optional (see Configuration)
```

`.env` is loaded on `import app.core.config`. **`RAG_BASE_URL` and `RAG_COLLECTION_BASE` are required** for `run_eval` (no hardcoded gateway defaults).

## Layout

```
app/
  core/           config.py, paths.py
  eval/           gold_dataset.py, run_eval.py, scoring.py, baseline.py, metadata.py,
                  llm_judge.py, dataset_version.py, supabase_client.py, supabase_store.py, supabase_cli.py
  http/           rag.py, inference.py
data_<env>/
  gold_dataset/   generated + hand-authored JSONL (gitignored; regenerate locally)
  report/         eval outputs (gitignored; use --summary-json / --report-json)
docs/eval.md      workflow + CLI reference
docs/version.md   dataset / ingest fingerprints + baseline pinning
docs/supabase.md  Supabase run history + DB baselines
supabase/schema.sql
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

## Quick start

Use `data_dev/…` or `data_prod/…` consistently (gold paths, report paths, and `--supabase-env` when recording to Supabase). For prod evals, point `.env` at the prod RAG gateway and collection (e.g. `RAG_COLLECTION_BASE=taixing_knowledge_prod`).

### Dev

```bash
# 1. Gold JSONL (default ingest root: ../layer-rag-ingest-v1/data_dev)
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

# 4. Optional: persist run to Supabase (SUPABASE_URL_DEV + SUPABASE_SECRET_KEY_DEV in .env)
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --limit 20 \
  --enable-llm-judge \
  --record-supabase \
  --supabase-env dev \
  --summary-json data_dev/report/rag_eval_summary.json

python -m app.eval.supabase_cli list-runs --env dev --limit 10
python -m app.eval.supabase_cli show-baseline --env dev
```

### Prod

```bash
# 1. Gold JSONL (pass prod ingest root explicitly)
python -m app.eval.gold_dataset \
  --data-roots ../layer-rag-ingest-v1/data_prod \
  --skip-consolidated-output \
  --split-output-dir data_prod/gold_dataset

# 2. Eval + reports (heuristic only)
python -m app.eval.run_eval \
  --gold data_prod/gold_dataset/easy_single_hop.jsonl \
  --summary-json data_prod/report/rag_eval_summary.json \
  --report-json data_prod/report/rag_eval_report.json

# 2b. LLM-as-judge
python -m app.eval.run_eval \
  --gold data_prod/gold_dataset/easy_single_hop.jsonl \
  --enable-llm-judge \
  --llm-judge-concurrency 10 \
  --limit 20 \
  --summary-json data_prod/report/rag_eval_summary.json \
  --report-json data_prod/report/rag_eval_report.json

# 3. Optional regression gate (pin a prod baseline JSON when ready)
python -m app.eval.run_eval \
  --gold data_prod/gold_dataset/easy_single_hop.jsonl \
  --limit 50 \
  --baseline-json tests/fixtures/baseline_summary.json \
  --baseline-tolerance 0.05

# 4. Optional: persist run to Supabase (SUPABASE_URL_PROD + SUPABASE_SECRET_KEY_PROD in .env)
python -m app.eval.run_eval \
  --gold data_prod/gold_dataset/easy_single_hop.jsonl \
  --limit 20 \
  --enable-llm-judge \
  --record-supabase \
  --supabase-env prod \
  --summary-json data_prod/report/rag_eval_summary.json

python -m app.eval.supabase_cli list-runs --env prod --limit 10
python -m app.eval.supabase_cli show-baseline --env prod
```

Full workflow: [docs/eval.md](docs/eval.md). Versioning and baseline pinning: [docs/version.md](docs/version.md). Supabase setup: [docs/supabase.md](docs/supabase.md).

## Configuration (`.env`)

### RAG + LLM judge

| Variable | Required | Purpose |
|----------|----------|---------|
| `RAG_BASE_URL` | **Yes** (for eval) | RAG gateway (no `/v1` suffix) |
| `RAG_COLLECTION_BASE` | **Yes** (for eval) | `collection_base` in API body |
| `LLM_JUDGE_URL` / `INFERENCE_URL` / `CHAT_BASE_URL` | With `--enable-llm-judge` | Chat API for semantic judge |
| `LLM_JUDGE_MODEL` / `CHAT_MODEL` | Optional | Judge model (default Qwen2.5-7B-Instruct) |
| `CHAT_BASE_URL` / `INFERENCE_URL` | With `--enable-must-contain-llm` | Gold generator LLM extraction |

### Supabase (optional — `--record-supabase`, `--baseline-supabase`, `--pin-baseline-supabase`)

Uses **supabase-py v2** (`create_client`), same pattern as `layer-gateway-api-v1`. Both HuntAI projects can live in one `.env`; `run_eval --supabase-env dev|prod` selects credentials and sets the row label in `rag_eval_*` tables.

**Dev vs prod on the CLI:** change together — `--gold` (`data_dev/…` vs `data_prod/…`), `--summary-json` / `--report-json`, and `--supabase-env` (`dev` vs `prod`). `--supabase-env` is inferred from `--gold` when omitted; set it explicitly for Supabase record/gate commands.

| Variable | Required | Purpose |
|----------|----------|---------|
| `SUPABASE_URL_DEV` | With Supabase + `--supabase-env dev` | HuntAI-dev project URL |
| `SUPABASE_SECRET_KEY_DEV` | With Supabase + `--supabase-env dev` | Dev secret key (`sb_secret_...` or legacy `service_role` JWT) |
| `SUPABASE_URL_PROD` | With Supabase + `--supabase-env prod` | HuntAI-prod project URL |
| `SUPABASE_SECRET_KEY_PROD` | With Supabase + `--supabase-env prod` | Prod secret key |

Legacy names still work: `SUPABASE_SERVICE_ROLE_KEY_DEV` / `_PROD`, or generic `SUPABASE_URL` + `SUPABASE_SECRET_KEY` as fallback. Do not use `anon` or `sb_publishable_...` for eval writes.

Example (from `.env.example`):

```bash
SUPABASE_URL_DEV=https://wgwcbcynjtmwgihohasw.supabase.co
SUPABASE_SECRET_KEY_DEV=sb_secret_...

SUPABASE_URL_PROD=https://oacomrvvnosgibkagoar.supabase.co
SUPABASE_SECRET_KEY_PROD=sb_secret_...
```

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

Supabase persistence is **local/ops only** (not wired in GitHub Actions yet). See [docs/supabase.md](docs/supabase.md) for `--record-supabase`, `--baseline-supabase`, and `--pin-baseline-supabase`.

## Related repos

- [layer-rag-ingest-v1](../layer-rag-ingest-v1) — ingest, synthetic questions, Qdrant upsert
- [layer-rag-query-v1](../layer-rag-query-v1) — RAG HTTP API contract
