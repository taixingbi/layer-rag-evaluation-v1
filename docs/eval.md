# RAG evaluation runbook

End-to-end workflow to build a gold dataset from ingested points, score the live RAG API, and write reports under `data_<env>/report/`.

**Scripts:** `rag_gold_eval/generate_gold_dataset.py`, `rag_gold_eval/run_eval.py`  
**Deep reference:** [gold-dataset.md](gold-dataset.md) (CLI flags, schema, scoring rules)  
**RAG API contract:** [layer-rag-query-v1/docs/eval.md](../../layer-rag-query-v1/docs/eval.md)

---

## What this repo evaluates

| Layer | Tool | Signals |
|-------|------|---------|
| **Retrieval** | `run_eval.py` | Rank, RR, MRR, Recall@k / Precision@k / NDCG@k / F1@k from `retrieval_hits` |
| **Answer** | `run_eval.py` | `must_contain`, citation `source` match, heuristic `quality_dimensions` |
| **Latency** | `run_eval.py` | `latency_ms` per row; p50/p95/p99 in summary |

For LLM-as-judge end-to-end metrics, use [layer-rag-evaluation-v1](../../layer-rag-evaluation-v1).

---

## Prerequisites

1. **Ingested KB** in Qdrant for the target env (`layer-rag-ingest-v1`: `./scripts/data1.sh dev` — see [data1.md](../../layer-rag-ingest-v1/docs/data1.md)).
2. **Synthetic questions** on points (`payload.synthetic_questions` non-empty). Without them, gold generation writes **0 rows** unless `--include-empty-questions`.
3. **RAG gateway** reachable (e.g. k3s NodePort `http://192.168.86.179:30183`, not necessarily localhost).
4. Env file (optional): `.env.dev` with `RAG_BASE_URL`, `RAG_COLLECTION_BASE`, `QDRANT_URL`, etc.

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

Required after synthetic-question enrichment so vectors and payload match what retrieval uses.

### 3. Generate gold JSONL

```bash
python3 rag_gold_eval/generate_gold_dataset.py \
  --data-roots data_dev \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset
```

Expect log: `rows_written` equal to your point count (one row per chunk by default), `invalid_single_hop=0`.

**Outputs:**

| File | Contents |
|------|----------|
| `data_dev/gold_dataset/easy_single_hop.jsonl` | One row per chunk (canonical question) |
| `data_dev/gold_dataset/paraphrase.jsonl` | Noisy variants (only if `--enable-noisy-queries`) |

Hand-authored splits (`multi_hop.jsonl`, `negative.jsonl`, etc.) are kept if already present; the generator only overwrites its split files.

### 4. Run eval + write reports

```bash
python3 rag_gold_eval/run_eval.py \
  --gold data_dev/gold_dataset/ \
  --rag-base-url http://192.168.86.179:30183 \
  --collection-base taixing_knowledge \
  --report-json data_dev/report/rag_eval_report.json \
  --summary-json data_dev/report/rag_eval_summary.json
```

Smoke test first:

```bash
python3 rag_gold_eval/run_eval.py \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --limit 10
```

---

## Gold row `id` vs `chunk_id`

| Field | Meaning | Used by eval for retrieval? |
|-------|---------|------------------------------|
| **`id`** | Qdrant **point UUID** (deterministic UUID5 at ingest; same value upserted to Qdrant) | **Yes** — matched against `retrieval_hits[].chunk_id` |
| **`chunk_id`** | Ingest ordinal inside document (e.g. `"0007"`) | **No** — informational only |

The RAG API sets `retrieval_hits[].chunk_id` from `hit.id` (Qdrant point id), not from `payload.chunk_id`.

After **re-chunk / re-prepare** (new `document_version` or chunk boundaries), point UUIDs can change. **Regenerate gold** from current `points_*.json` before eval, or retrieval metrics will show misses despite correct content.

---

## Report files

### `rag_eval_summary.json` (aggregates)

Key fields:

| Field | Meaning |
|-------|---------|
| `mrr_retrieve` / `mrr_rerank` | Mean reciprocal rank over all scored rows |
| `mean_rr_retrieve_when_found` | MRR over rows where gold chunk was found (excludes misses) |
| `recall_at_5_retrieve` | Fraction of rows with gold in top-5 at retrieve stage |
| `must_contain_pass` / `must_contain_total` | Answer substring checks |
| `quality_score_mean` | Heuristic answer quality (5 binary dims) |
| `latency_ms_p50` / `p95` / `p99` | End-to-end latency |
| `rag_calls_failed` | HTTP/API errors (should be 0) |

### `rag_eval_report.json` (per row)

Array of one object per gold row: `question`, `answer_preview`, `must_contain_pass`, `rank_retrieve`, `rank_rerank`, `rr_retrieve`, `hit_retrieve_at_5`, retrieval fields, `quality_dimensions`, errors, etc. Use this to debug individual failures.

Without `--summary-json` / `--report-json`, only the summary JSON is printed to **stdout**.

---

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `rows_written=0` | Points have empty `synthetic_questions` | Run `synthetic_questions.py`, then regenerate gold |
| Gold written to repo root | Ran generator with **default** flags (no `--split-output-dir`) | Use explicit `--data-roots data_dev --split-output-dir data_dev/gold_dataset` |
| `rows_written=N` but `env: prod` | Default `--data-roots` includes `data_prod`; dev had no synthetic Q | Pass `--data-roots data_dev` only |
| Retrieval misses, live curl works | Stale gold `id` after re-ingest | Regenerate gold from current `points_*.json` |
| `rag_calls_failed` > 0 | Wrong `--rag-base-url` or gateway down | Use cluster NodePort; check `RAG_BASE_URL` |
| `retrieval_eval_skipped` | Gold `id` is not a UUID (multi-hop synthetic ids) | Expected; those rows skip retrieval metrics |

---

## Environment variables

| Variable | Used by |
|----------|---------|
| `RAG_BASE_URL` | `run_eval.py` default `--rag-base-url` |
| `RAG_COLLECTION_BASE` | `run_eval.py` default `--collection-base` |
| `CHAT_BASE_URL` / `CHAT_MODEL` | `generate_gold_dataset.py` with `--enable-must-contain-llm` |
| `QDRANT_URL` / `COLLECTION_NAME` | `layer-rag-ingest-v1` `upsert_qdrant.py` |

Collection name resolves to `<COLLECTION_NAME>_<env>` when `ENV=dev|qa|prod`.

---

## Suggested regression cadence

1. `layer-rag-ingest-v1`: `./scripts/data1.sh dev` (or upsert only if points unchanged except synthetic Q).
2. Regenerate gold → `data_dev/gold_dataset/`.
3. `run_eval.py` with `--report-json` + `--summary-json`.
4. Compare `mrr_rerank`, `recall_at_5_rerank`, `must_contain_pass` rate, and `latency_ms_p95` across runs.
5. Pin in notes: ingest manifest / git SHA, `collection_base`, `k`, `k_max`, embed model.

---

## Related docs

- [gold-dataset.md](gold-dataset.md) — generator + `run_eval.py` CLI and scoring rules
- [layer-rag-ingest-v1/docs/data1.md](../../layer-rag-ingest-v1/docs/data1.md) — ingest pipeline
- [layer-rag-ingest-v1/docs/identity-key.md](../../layer-rag-ingest-v1/docs/identity-key.md) — how point UUIDs are derived
- [layer-rag-query-v1/docs/eval.md](../../layer-rag-query-v1/docs/eval.md) — HTTP eval settings and `retrieval_hits` shape
