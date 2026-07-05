# Supabase persistence for RAG eval

Store **eval run history** and **pinned baselines** in Supabase Postgres. Gold JSONL stays on disk; only summaries and fingerprints go to the DB.

**Schema:** [../supabase/schema.sql](../supabase/schema.sql)  
**Code:** `app/eval/supabase_client.py`, `app/eval/supabase_store.py`, `app/eval/supabase_cli.py`

---

## Tables

| Table | Purpose |
|-------|---------|
| `rag_eval_runs` | One row per `run_eval` (metrics + `run_meta` + full `summary_json`) |
| `rag_eval_baselines` | Named, active baseline per env for regression gates |

See your Supabase schema diagram for column details.

---

## Environment (`.env`)

Both HuntAI projects can live in one file. ``run_eval --supabase-env dev|prod`` selects credentials and sets the row label in ``rag_eval_*`` tables.

```bash
# HuntAI-dev
SUPABASE_URL_DEV=https://wgwcbcynjtmwgihohasw.supabase.co
SUPABASE_SECRET_KEY_DEV=sb_secret_...

# HuntAI-prod
SUPABASE_URL_PROD=https://oacomrvvnosgibkagoar.supabase.co
SUPABASE_SECRET_KEY_PROD=sb_secret_...
```

Eval persistence only needs **URL + secret key** per env. You do not need `SUPABASE_PUBLISHABLE_KEY_*` here.

**Legacy names** still work per env or as generic fallback:

```bash
SUPABASE_SERVICE_ROLE_KEY_DEV=sb_secret_...
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SECRET_KEY=sb_secret_...
```

Secret key values: `sb_secret_...` (Settings → **API Keys**) or legacy `service_role` JWT `eyJ...` (Settings → **API**).

Uses **`supabase-py` v2** (`create_client`) — same pattern as `layer-gateway-api-v1`.

---

## Dev vs prod (keep in sync)

Both HuntAI projects are configured in `.env` (`SUPABASE_URL_DEV` / `SUPABASE_SECRET_KEY_DEV` and `_PROD` variants). **`run_eval --supabase-env dev|prod`** picks credentials and sets the row label in `rag_eval_*` tables.

When switching from dev to prod, change **these together** on every Supabase eval command:

| Flag / path | Dev | Prod |
|-------------|-----|------|
| `--gold` | `data_dev/gold_dataset/...` | `data_prod/gold_dataset/...` |
| `--summary-json` / `--report-json` | `data_dev/report/...` | `data_prod/report/...` |
| `--supabase-env` | `dev` | `prod` |

`--supabase-env` is **inferred from `--gold`** when omitted (`data_dev/…` → `dev`, `data_prod/…` → `prod`), but set it explicitly when recording or gating. Match **`supabase_cli --env`** when querying history.

---

## Record a run

After eval, insert into `rag_eval_runs`:

```bash
# Dev
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --enable-llm-judge \
  --summary-json data_dev/report/rag_eval_summary.json \
  --record-supabase \
  --supabase-env dev \
  --supabase-notes "weekly dev regression"

# Prod — swap --gold, --summary-json, --supabase-env (see Dev vs prod above)
python -m app.eval.run_eval \
  --gold data_prod/gold_dataset/easy_single_hop.jsonl \
  --enable-llm-judge \
  --summary-json data_prod/report/rag_eval_summary.json \
  --record-supabase \
  --supabase-env prod \
  --supabase-notes "weekly prod regression"
```

`pass` is set from baseline checks (file and/or Supabase) in the same invocation.

---

## Pin an active baseline

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/ \
  --enable-llm-judge \
  --summary-json data_dev/report/rag_eval_summary.json \
  --pin-baseline-supabase dev-2026-07-04 \
  --supabase-env dev

# Prod: data_prod/gold_dataset/, data_prod/report/..., --supabase-env prod
```

Deactivates previous `active=true` baselines for that env and inserts a new active row with metrics + `baseline_json` fingerprints.

---

## Gate against Supabase baseline

```bash
python -m app.eval.run_eval \
  --gold data_dev/gold_dataset/ \
  --enable-llm-judge \
  --baseline-supabase \
  --supabase-env dev \
  --record-supabase \
  --baseline-tolerance 0.05

# Prod: data_prod/gold_dataset/, --supabase-env prod
```

Optional: `--baseline-supabase-name my-baseline` instead of active baseline.

Combine with file baseline: `--baseline-json tests/fixtures/baseline_summary.json` **and** `--baseline-supabase`.

---

## Query history (CLI)

```bash
python -m app.eval.supabase_cli list-runs --env dev --limit 10
python -m app.eval.supabase_cli show-baseline --env dev

python -m app.eval.supabase_cli list-runs --env prod --limit 10
python -m app.eval.supabase_cli show-baseline --env prod
```

---

## What is not in Supabase

| Data | Where |
|------|--------|
| Gold JSONL rows | `data_<env>/gold_dataset/` (files) |
| Full per-row report | Local path in `report_storage_path` (upload to Storage optional later) |

---

## Related

- [version.md](version.md) — `gold_dataset_sha256`, ingest manifest hashing
- [eval.md](eval.md) — full eval workflow
