# RAG evaluation

End-to-end workflow and CLI reference: ingest KB → generate gold JSONL → score the live RAG API → write reports under `data_<env>/report/`.

**Scripts:** `python -m app.eval.gold_dataset`, `python -m app.eval.run_eval`  
**Code:** `app/eval/gold_dataset.py`, `app/eval/run_eval.py`  
**Points input:** [layer-rag-ingest-v1](../../layer-rag-ingest-v1) `data_<env>/data1/processed/points_*.json`  
**Outputs:** `data_<env>/gold_dataset/` and `data_<env>/report/` in this repo  
**RAG HTTP contract:** [layer-rag-query-v1/docs/eval.md](../../layer-rag-query-v1/docs/eval.md)

---

## What this repo evaluates

| Layer | Module | Signals |
|-------|--------|---------|
| **Retrieval** | `app.eval.run_eval` | Rank, RR, MRR, Recall@k / Precision@k / NDCG@k / F1@k from `retrieval_hits` |
| **Answer** | `app.eval.run_eval` | `must_contain`, citation `source`, **`heuristic_quality_*`** (proxy rules, not LLM judge) |
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
3. **RAG gateway** reachable (e.g. `http://192.168.86.179:30183`).
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
  --rag-base-url http://192.168.86.179:30183 \
  --collection-base taixing_knowledge \
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
4. **`retrieval_hits`** — Gold UUID `id` vs `retrieval_hits[].chunk_id` in `retrieve` / `rerank` → rank, RR, MRR, Recall@k, Precision@k, NDCG@k, F1@k (`--recall-at-k`, default `5,10,40`). Non-UUID ids → `retrieval_eval_skipped`.
5. **`heuristic_quality`** — Proxy dimensions (`correct`, `faithful`, …) derived from `must_contain` + citations; summary uses `heuristic_quality_*` keys (not semantic LLM/human labels).

Requests use `collection_base`, `k`, `k_max`. Correlation ids in **headers** (`X-Request-Id`, `X-Session-Id`). Body: `stream: false`, `expand_on_not_found: false`, `include_follow_up_questions: false`.

### `run_eval` CLI

| Flag | Default | Role |
|------|---------|------|
| `--gold` | (required) | JSONL file or directory of `*.jsonl` |
| `--rag-base-url` | `RAG_BASE_URL` or `http://192.168.86.179:30183` | Gateway base (no `/v1`) |
| `--collection-base` | `RAG_COLLECTION_BASE` or `taixing_knowledge` | `collection_base` in body |
| `--k` / `--k-max` | `5` / `40` | Retrieval params |
| `--concurrency` | `40` | Max concurrent RAG requests |
| `--limit` | `0` | Max rows (`0` = all) |
| `--skip-retrieval-hits` | off | Skip retrieval metrics |
| `--recall-at-k` | `5,10,40` | k values for @k metrics |
| `--report-json` | off | Per-row JSON array |
| `--baseline-json` | off | Fail if metrics drop below pinned summary by `--baseline-tolerance` |
| `--baseline-tolerance` | `0.05` | Allowed drop for higher-is-better metrics |

By default only **stdout** (JSON summary). Reports under `data_<env>/report/` are gitignored; pass `--summary-json` / `--report-json` to persist locally.

### `run_eval` examples

```bash
python -m app.eval.run_eval --gold data_dev/gold_dataset/

python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/paraphrase.jsonl \
  --report-json data_dev/report/rag_eval_paraphrase_report.json \
  --summary-json data_dev/report/rag_eval_paraphrase_summary.json
```

Summary includes `mrr_*`, `recall_at_*`, `latency_ms_*`, `must_contain_*`, `heuristic_quality_*`, `errors_sample`.

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
| `mrr_retrieve` / `mrr_rerank` | Mean reciprocal rank |
| `recall_at_5_*` | Gold in top-5 |
| `must_contain_pass` / `must_contain_total` | Substring checks |
| `heuristic_quality_score_mean` | Proxy answer quality (not LLM judge) |
| `latency_ms_p50` / `p95` / `p99` | Latency |
| `rag_calls_failed` | HTTP errors (should be 0) |

### `rag_eval_report.json`

Per-row debug: ranks, @k hits, `heuristic_quality`, answer preview, errors.

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

Exits non-zero if key metrics (e.g. `recall_at_5_rerank`, `mrr_rerank`) drop more than tolerance below baseline.

---

## Development & CI

```bash
pip install -e ".[dev]"
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs tests on push/PR to `main` without calling live RAG.

---

## Regression cadence

1. Ingest / upsert ([data1.sh](../../layer-rag-ingest-v1/scripts/data1.sh) or upsert only).
2. Regenerate gold → `data_dev/gold_dataset/`.
3. `run_eval` with `--report-json` + `--summary-json`.
4. Track `mrr_rerank`, `recall_at_5_rerank`, `must_contain_pass`, `latency_ms_p95`.
5. Pin ingest SHA, `collection_base`, `k`, `k_max`, embed model.

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
