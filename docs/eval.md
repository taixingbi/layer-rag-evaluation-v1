# RAG evaluation

End-to-end workflow and CLI reference: ingest KB → generate gold JSONL → score the live RAG API → write reports under `data_<env>/report/`.

**Scripts:** `python -m app.eval.gold_dataset`, `python -m app.eval.run_eval`  
**Code:** `app/eval/gold_dataset.py`, `app/eval/run_eval.py`  
**Points input:** [layer-rag-ingest-v1](../../layer-rag-ingest-v1) `data_<env>/data1/processed/points_*.json`  
**Outputs:** `data_<env>/gold_dataset/` and `data_<env>/report/` in this repo  
**Versioning:** [version.md](version.md) — gold + ingest fingerprints, baseline pinning  
**RAG HTTP contract:** [layer-rag-query-v1/docs/eval.md](../../layer-rag-query-v1/docs/eval.md)

---

## What this repo evaluates

| Layer | Module | Signals |
|-------|--------|---------|
| **Retrieval** | `app.eval.run_eval` | Rank, RR, MRR, Recall@k / Precision@k / NDCG@k / F1@k from `retrieval_hits` |
| **Answer (heuristic)** | `app.eval.run_eval` | `must_contain`, citation `source`, `heuristic_quality_*` (fast proxy) |
| **Answer (LLM judge)** | `app.eval.run_eval --enable-llm-judge` | `llm_judge_*` semantic scores in the same run |
| **Latency** | `app.eval.run_eval` | `latency_ms` per row; p50 / p95 / p99 in summary |

---

## Commands

| Command | Purpose |
|---------|---------|
| `python -m app.eval.gold_dataset` | Build gold JSONL from ingest points |
| `python -m app.eval.run_eval` | `POST /v1/rag/query` per row; score retrieval + answers |

---

## Prerequisites

1. **Ingested KB** in Qdrant — [layer-rag-ingest-v1](../../layer-rag-ingest-v1): `./scripts/data1.sh dev` ([data1.md](../../layer-rag-ingest-v1/docs/data1.md)).
2. **Synthetic questions** on points (`payload.synthetic_questions` non-empty). Without them, gold generation writes **0 rows** unless `--include-empty-questions`.
3. **RAG gateway** reachable at `RAG_BASE_URL` (see `.env.example`).
4. **`.env`** with **`RAG_BASE_URL`** and **`RAG_COLLECTION_BASE`** (required; copy from `.env.example`).

---

## End-to-end workflow (dev)

Run from **layer-rag-evaluation-v1** repo root.

### 1. Synthetic questions (if missing)

```bash
python3 ../layer-rag-ingest-v1/app/synthetic_questions.py \
  --data-dir ../layer-rag-ingest-v1/data_dev/data1/processed \
  --questions-per-chunk 3
```

Updates `synthetic_questions`, `embed_text`, and `embed_token_count`. Point **`id` is unchanged**.

### 2. Re-upsert to Qdrant

```bash
cd ../layer-rag-ingest-v1
set -a && source .env.dev && set +a
python3 app/upsert_qdrant.py \
  --data-dir data_dev/data1/processed \
  --pattern "points_*.json"
```

Required after synthetic-question enrichment so vectors and payload match retrieval.

### 3. Generate gold JSONL

```bash
cd ../layer-rag-evaluation-v1
python -m app.eval.gold_dataset \
  --data-roots ../layer-rag-ingest-v1/data_dev \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset
```

Expect log: `rows_written` ≈ point count, `invalid_single_hop=0`.

**Outputs:**

| File | Contents |
|------|----------|
| `easy_single_hop.jsonl` | One row per chunk (canonical synthetic question) |
| `paraphrase.jsonl` | Noisy variants (only with `--enable-noisy-queries`) |
| `multi_hop.jsonl`, `nagative.jsonl`, … | Hand-authored; generator does not overwrite |

### 4. Run eval + write reports

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/ \
  --report-json data_dev/report/rag_eval_report.json \
  --summary-json data_dev/report/rag_eval_summary.json
```

Smoke test first:

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --limit 10
```

---

## Generator (`app.eval.gold_dataset`)

### Behavior

- Scans `--data-roots` (default: sibling ingest **`data_dev` only**; pass explicit roots for qa/prod).
- Finds `**/processed/points_*.json` (override with `--glob`).
- Reads `payload.synthetic_questions`; uses `payload.text` as `answer` / `text`.
- Emits one JSONL row per question variant (canonical + optional noisy).
- **`must_contain`:** heuristic by default; `--enable-must-contain-llm` uses chat API (`--llm-concurrency`, default **40**).

### Output schema (each JSONL line)

- `env`, `source_file`, `id`, `question`, `answer`, `must_contain`
- `source`, `doc_type`, `section`, `chunk_id`, `text`
- `case_type`, `required_sources`, `expected_behavior`, `query_type`, `eval_bucket`

### Generator CLI

| Flag | Role |
|------|------|
| `--data-roots` | Ingest roots (default: sibling `data_dev` only) |
| `--glob` | Glob under each root (default: `**/processed/points_*.json`) |
| `--output` | Consolidated JSONL (default: `gold_dataset.jsonl`) |
| `--skip-consolidated-output` | Omit consolidated file; requires `--split-output-dir` |
| `--split-output-dir` | Directory for split JSONL files |
| `--include-empty-questions` | Row with empty question if no synthetic questions |
| `--no-dedup` | Keep duplicate `(env, id, question)` rows |
| `--enable-must-contain-llm` | LLM `must_contain` extraction |
| `--enable-noisy-queries` | Noisy query variants |
| `--max-paraphrases-per-fact` | Max variants per fact (default `3`) |
| `--chat-base-url`, `--chat-model`, `--chat-api-key` | Chat API for LLM must_contain |
| `--llm-concurrency` | Max concurrent LLM calls (default **40**) |

### Generator examples

```bash
# LLM must_contain + paraphrases
python -m app.eval.gold_dataset \
  --data-roots ../layer-rag-ingest-v1/data_dev \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset \
  --enable-must-contain-llm \
  --enable-noisy-queries \
  --max-paraphrases-per-fact 2

# Consolidated + splits
python -m app.eval.gold_dataset \
  --data-roots ../layer-rag-ingest-v1/data_dev \
  --output data_dev/gold_dataset/gold_dataset.jsonl \
  --split-output-dir data_dev/gold_dataset
```

---

## RAG evaluation (`app.eval.run_eval`)

Runs **`POST {rag_base_url}/v1/rag/query`** per gold row, then scores the response.

### Scoring rules

1. **`must_contain`** — Each fragment in RAG `answer` (case-insensitive). Empty list skips this axis.
2. **`source` (single-hop)** — Gold `source` must appear in a citation (unless `multi` / `negative`).
3. **`required_sources` (multi-hop)** — Every listed source in citations.
4. **`retrieval_hits`** — Gold UUID `id` vs `retrieval_hits[].chunk_id` in `retrieve` / `rerank` → rank, RR, **MRR**, **Recall@k** (primary), Precision@k, NDCG@k, F1@k. Each row has **one relevant document**; Precision@k = `1/k` when the gold chunk is in top-k (not multi-source precision). Non-UUID ids → `retrieval_eval_skipped`.
5. **`heuristic_quality`** — Proxy dimensions (`correct`, `faithful`, …) from `must_contain` + citations; always computed.
6. **`llm_judge`** (optional, `--enable-llm-judge`) — LLM scores the same five dimensions semantically vs gold reference answer; per-row `llm_judge`, `llm_judge_score`, `llm_judge_reason`.

Requests use `collection_base`, `k`, `k_max`. Correlation ids in **headers** (`X-Request-Id`, `X-Session-Id`). Body: `stream: false`, `expand_on_not_found: false`, `include_follow_up_questions: false`.

### `run_eval` CLI

| Flag | Default | Role |
|------|---------|------|
| `--gold` | (required) | JSONL file or directory of `*.jsonl` |
| `--rag-base-url` | `RAG_BASE_URL` (required if unset on CLI) | Gateway base (no `/v1`) |
| `--collection-base` | `RAG_COLLECTION_BASE` (required if unset on CLI) | `collection_base` in body |
| `--k` / `--k-max` | `5` / `40` | Retrieval params |
| `--concurrency` | `40` | Max concurrent RAG requests |
| `--limit` | `0` | Max rows (`0` = all) |
| `--skip-retrieval-hits` | off | Skip retrieval metrics |
| `--recall-at-k` | `5,10,40` | k values for @k metrics |
| `--report-json` | off | Per-row JSON array |
| `--baseline-json` | off | Fail if metrics drop below pinned summary by `--baseline-tolerance` |
| `--baseline-tolerance` | `0.05` | Allowed drop for higher-is-better metrics |
| `--enable-llm-judge` | off | LLM semantic answer scoring (same script/run) |
| `--llm-judge-base-url` | `LLM_JUDGE_URL` / `INFERENCE_URL` / `CHAT_BASE_URL` | Chat API for judge |
| `--llm-judge-model` | `LLM_JUDGE_MODEL` / `CHAT_MODEL` | Judge model name |
| `--llm-judge-concurrency` | `10` | Max concurrent judge requests |
| `--ingest-manifest` | auto from `data_<env>` | Ingest manifest path for `ingest_manifest_sha256` in `run_meta` |

By default only **stdout** (JSON summary). Reports under `data_<env>/report/` are gitignored; pass `--summary-json` / `--report-json` to persist locally.

### LLM-as-judge example

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --enable-llm-judge \
  --llm-judge-concurrency 10 \
  --limit 20 \
  --summary-json data_dev/report/rag_eval_summary.json
```

Heuristic metrics are always computed; LLM judge adds `llm_judge_*` summary keys when enabled. Start with `--limit` — full corpus = one LLM call per row.

### `run_eval` examples

```bash
python -m app.eval.run_eval --gold data_dev/gold_dataset/

python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/paraphrase.jsonl \
  --report-json data_dev/report/rag_eval_paraphrase_report.json \
  --summary-json data_dev/report/rag_eval_paraphrase_summary.json
```

Summary includes `run_meta`, `mrr_*`, `recall_at_*`, `latency_ms_*`, `must_contain_*`, `heuristic_quality_*`, `llm_judge_*` (when enabled), `errors_sample`.

---

## Gold row `id` vs `chunk_id`

| Field | Meaning | Used for retrieval scoring? |
|-------|---------|-------------------------------|
| **`id`** | Qdrant point UUID (UUID5 at ingest) | **Yes** — vs `retrieval_hits[].chunk_id` |
| **`chunk_id`** | Ingest ordinal (e.g. `"0007"`) | **No** |

After re-chunk / re-ingest, **regenerate gold** or retrieval metrics will miss.

---

## Report files

### `rag_eval_summary.json`

| Field | Meaning |
|-------|---------|
| `run_meta` | Git SHA, package version, k/k_max, collection, gold paths, row counts, **`gold_dataset_sha256`**, **`ingest_manifest_sha256`** (when manifest found), UTC timestamp |
| `mrr_retrieve` / `mrr_rerank` | Mean reciprocal rank (**primary retrieval**) |
| `recall_at_5_*` | Gold chunk in top-k (**primary retrieval**) |
| `precision_at_5_*` | Single-doc precision (`1/k` when hit; not multi-source) |
| `must_contain_pass` / `must_contain_scored_rows` | Substring checks |
| `must_contain_pass_rate` | Fraction of scored rows passing all fragments |
| `heuristic_quality_score_mean` | Proxy answer quality (always) |
| `llm_judge_score_mean` | LLM semantic quality (with `--enable-llm-judge`) |
| `latency_ms_p50` / `p95` / `p99` | Latency |
| `rag_calls_failed` | HTTP errors (should be 0) |

### `rag_eval_report.json`

Per-row debug: ranks, @k hits, `heuristic_quality`, `llm_judge`, `llm_judge_reason`, answer preview, errors.

---

## Validation checklist

**Generator:** `invalid_single_hop=0`, non-zero `easy_single_hop.jsonl`, spot-check `id`/`question`/`answer` vs points.

**Eval:** `rag_calls_failed=0`, `must_contain_pass` and retrieval metrics meet bar, latency p95 within SLO.

---

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `rows_written=0` | Empty `synthetic_questions` | Run `synthetic_questions.py`, regenerate gold |
| Skipping `.../data_qa` | Default roots include all envs | `--data-roots ../layer-rag-ingest-v1/data_dev` only |
| Prod rows on dev run | Default includes `data_prod` | Restrict `--data-roots` |
| Retrieval misses, curl OK | Stale gold `id` | Regenerate gold from current points |
| `rag_calls_failed` > 0 | Bad URL or ids in JSON body | Check `RAG_BASE_URL`; use headers for correlation ids |
| `retrieval_eval_skipped` | Non-UUID gold `id` | Expected for hand-authored multi-hop |

---

## Environment variables

| Variable | Used by |
|----------|---------|
| `RAG_BASE_URL` | `run_eval` default `--rag-base-url` |
| `RAG_COLLECTION_BASE` | `run_eval` default `--collection-base` |
| `LLM_JUDGE_URL` / `INFERENCE_URL` / `CHAT_BASE_URL` | `run_eval --enable-llm-judge` |
| `LLM_JUDGE_MODEL` / `CHAT_MODEL` | Judge model name |
| `CHAT_BASE_URL` / `INFERENCE_URL` | `gold_dataset --enable-must-contain-llm` |
| `QDRANT_URL` / `COLLECTION_NAME` / `ENV` | **layer-rag-ingest-v1** upsert |

Collection: `<COLLECTION_NAME>_<env>` when `ENV=dev|qa|prod`.

---

## Regression baseline

Pin a summary JSON from a known-good run, then gate future evals:

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --baseline-json tests/fixtures/baseline_summary.json \
  --baseline-tolerance 0.05
```

Exits non-zero if key metrics (e.g. `recall_at_5_rerank`, `mrr_rerank`, `llm_judge_score_mean`) drop more than tolerance below baseline.

When the baseline JSON includes **`gold_dataset_sha256`** or **`ingest_manifest_sha256`** (copy from a known-good `run_meta`), the gate also fails if the current eval used different gold or ingest inputs. Details: [version.md](version.md).

Pin a baseline after a good run:

```bash
# Copy metrics + run_meta fingerprints from rag_eval_summary.json into tests/fixtures/baseline_summary.json
jq '{rag_calls_failed, mrr_rerank, recall_at_5_rerank, llm_judge_score_mean,
     gold_dataset_sha256: .run_meta.gold_dataset_sha256,
     ingest_manifest_sha256: .run_meta.ingest_manifest_sha256,
     collection_base: .run_meta.collection_base}' \
  data_dev/report/rag_eval_summary.json
```

---

## Development & CI

```bash
pip install -e ".[dev]"
pytest
ruff check app tests
```

GitHub Actions (`.github/workflows/ci.yml`) runs unit tests on push/PR to `main` without calling live RAG.

### Repository secrets (live eval workflow)

`.github/workflows/rag-eval.yml` reads:

| Secret | Required | Purpose |
|--------|----------|---------|
| `RAG_BASE_URL` | Yes | RAG gateway (no `/v1` suffix) |
| `RAG_COLLECTION_BASE` | Yes | `collection_base` for eval requests |

Add both under **Settings → Secrets and variables → Actions**. The workflow validates they are set before running.

Scheduled smoke: weekly, one-row fixture (`tests/fixtures/gold_single_row.jsonl`). Full-folder regression with `--baseline-json` remains a local or self-hosted job (gold JSONL is gitignored).

---

## Regression cadence

1. Ingest / upsert ([data1.sh](../../layer-rag-ingest-v1/scripts/data1.sh) or upsert only).
2. Regenerate gold → `data_dev/gold_dataset/`.
3. `run_eval` with `--report-json` + `--summary-json`.
4. Track `mrr_rerank`, `recall_at_5_rerank`, `llm_judge_score_mean` (when judge enabled), `latency_ms_p95`.
5. Pin **`gold_dataset_sha256`** and **`ingest_manifest_sha256`** from `run_meta` in baseline JSON.
6. Pin ingest SHA, `collection_base`, `k`, `k_max`, embed model.

---

## Notes

- `env` = ingest folder with `data_` stripped (`data_dev` → `dev`).
- Dedup key: `(env, id, question)`.
- Heuristic `must_contain` cached per `(env, point id)` before paraphrase rows.

---

## Related docs

- [layer-rag-ingest-v1/docs/data1.md](../../layer-rag-ingest-v1/docs/data1.md) — ingest pipeline
- [layer-rag-ingest-v1/docs/identity-key.md](../../layer-rag-ingest-v1/docs/identity-key.md) — point UUID derivation
- [layer-rag-query-v1/docs/eval.md](../../layer-rag-query-v1/docs/eval.md) — RAG HTTP eval settings
