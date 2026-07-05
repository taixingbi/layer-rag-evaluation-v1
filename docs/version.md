# Dataset and eval run versioning

How `run_eval` fingerprints gold data and ingest inputs so regression runs are comparable. See also [eval.md](eval.md) for the full workflow.

**Code:** `app/eval/dataset_version.py`, `app/eval/metadata.py`, `app/eval/baseline.py`

---

## Why versioning matters

Gold JSONL under `data_<env>/gold_dataset/` is **gitignored** and regenerated from [layer-rag-ingest-v1](../../layer-rag-ingest-v1) points. Two evals with the same `--gold` path can use **different content** after:

- Re-ingest or re-chunk (Qdrant point UUIDs change)
- Regenerating gold with `app.eval.gold_dataset`
- Editing hand-authored rows (`multi_hop.jsonl`, etc.)

Production regression should pin **what was evaluated**, not only metric thresholds.

---

## What gets recorded

Every `run_eval` summary includes a `run_meta` block. Version-related fields:

| Field | Source | Meaning |
|-------|--------|---------|
| `eval_package_version` | Package metadata | Eval harness version |
| `git_sha` | Eval repo git | Commit of `layer-rag-evaluation-v1` |
| `gold_paths` | CLI `--gold` | Resolved JSONL file paths |
| `gold_rows_loaded` | Loaded rows | Before `--limit` |
| `gold_rows_evaluated` | Result count | After `--limit` |
| `gold_dataset_files` | Content hash | Per-file `path`, `sha256`, `bytes` |
| `gold_dataset_sha256` | Combined hash | Fingerprint of the full gold set used |
| `ingest_manifest_path` | Auto or CLI | Path to ingest manifest JSON |
| `ingest_manifest_sha256` | Manifest file | Fingerprint of ingest snapshot |
| `ingest_env`, `ingest_run_id`, … | Manifest JSON | Selected ingest metadata when parseable |
| `collection_base` | Env / CLI | RAG collection evaluated |
| `k`, `k_max`, `recall_at_k` | CLI | Retrieval params (must match across runs) |
| `enable_llm_judge`, `llm_judge_model` | CLI | Judge config when used |

Example (abbreviated):

```json
"run_meta": {
  "gold_dataset_sha256": "a1b2c3…",
  "gold_dataset_files": [
    {"path": "…/easy_single_hop.jsonl", "sha256": "…", "bytes": 123456}
  ],
  "gold_rows_loaded": 196,
  "gold_rows_evaluated": 196,
  "ingest_manifest_sha256": "d4e5f6…",
  "ingest_manifest_path": "…/layer-rag-ingest-v1/data_dev/data1/processed/ingest_manifest_latest.json",
  "ingest_env": "dev",
  "collection_base": "taixing_knowledge"
}
```

---

## How hashes are computed

### Gold dataset (`gold_dataset_sha256`)

1. Resolve all `*.jsonl` files from `--gold` (files or directories).
2. SHA-256 each file’s **raw bytes**.
3. Combine digests in **stable sorted path order** into one dataset hash.

Changing any line in any included JSONL changes `gold_dataset_sha256`.

### Ingest manifest (`ingest_manifest_sha256`)

SHA-256 of the manifest file bytes. Typically:

```
../layer-rag-ingest-v1/data_<env>/data1/processed/ingest_manifest_latest.json
```

`<env>` is inferred from gold paths (`data_dev/…` → `dev`).

Override with:

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/ \
  --ingest-manifest ../layer-rag-ingest-v1/data_dev/data1/processed/ingest_manifest_latest.json
```

If no manifest is found, ingest fields are omitted (gold hash still recorded).

---

## Pinning a baseline

After a known-good eval, copy metrics **and** fingerprints into `tests/fixtures/baseline_summary.json` (or your own baseline file):

```bash
jq '{
  description: "Pinned dev eval YYYY-MM-DD",
  rag_calls_failed,
  mrr_rerank,
  recall_at_5_rerank,
  llm_judge_score_mean,
  gold_dataset_sha256: .run_meta.gold_dataset_sha256,
  ingest_manifest_sha256: .run_meta.ingest_manifest_sha256,
  collection_base: .run_meta.collection_base
}' data_dev/report/rag_eval_summary.json
```

Gate future runs:

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/ \
  --baseline-json tests/fixtures/baseline_summary.json \
  --baseline-tolerance 0.05 \
  --summary-json data_dev/report/rag_eval_summary.json
```

The gate fails when:

1. **Metrics** drop more than `--baseline-tolerance` below pinned values (`mrr_rerank`, `recall_at_5_rerank`, `llm_judge_score_mean`, …).
2. **`rag_calls_failed`** exceeds baseline.
3. **`gold_dataset_sha256`**, **`ingest_manifest_sha256`**, or **`collection_base`** differ from baseline (when present in baseline JSON).

---

## When to refresh the baseline

| Event | Action |
|-------|--------|
| Intentional KB / ingest update | Regenerate gold → re-run eval → **update baseline** fingerprints + metrics |
| RAG config change (`k`, `collection_base`) | Re-run eval → update baseline (fingerprints may be unchanged) |
| Eval harness code change only | Metrics may shift slightly; re-pin if acceptable |
| Accidental gold regen | Compare `gold_dataset_sha256`; baseline gate catches drift |

---

## Versioning checklist

1. Ingest + upsert ([layer-rag-ingest-v1](../../layer-rag-ingest-v1/docs/data1.md)).
2. Generate gold → `data_<env>/gold_dataset/`.
3. Run eval with `--summary-json` (and `--enable-llm-judge` when judging answers).
4. Save `run_meta.gold_dataset_sha256` and `run_meta.ingest_manifest_sha256`.
5. Pin baseline JSON with metrics + fingerprints.
6. On every regression run, use the **same** `--gold` corpus and compare against baseline.

---


## Related docs

- [eval.md](eval.md) — CLI reference and workflow
- [layer-rag-ingest-v1/docs/identity-key.md](../../layer-rag-ingest-v1/docs/identity-key.md) — Qdrant point UUID derivation (gold `id` must match)
