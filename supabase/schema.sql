-- RAG eval history (Supabase / Postgres)
-- Apply in Supabase SQL editor if tables are not created yet.

create table if not exists public.rag_eval_baselines (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  env text not null,
  name text not null,
  active boolean not null default true,
  collection_base text,
  gold_dataset_sha256 text,
  ingest_manifest_sha256 text,
  mrr_rerank double precision,
  recall_at_5_rerank double precision,
  llm_judge_score_mean double precision,
  rag_calls_failed integer,
  baseline_json jsonb,
  notes text
);

create index if not exists rag_eval_baselines_env_active_idx
  on public.rag_eval_baselines (env, active, created_at desc);

create unique index if not exists rag_eval_baselines_env_name_idx
  on public.rag_eval_baselines (env, name);

create table if not exists public.rag_eval_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  env text not null,
  collection_base text,
  gold_dataset_sha256 text,
  ingest_manifest_sha256 text,
  rows_loaded integer,
  rows_evaluated integer,
  rag_calls_failed integer,
  mrr_rerank double precision,
  recall_at_5_rerank double precision,
  precision_at_5_rerank double precision,
  ndcg_at_5_rerank double precision,
  llm_judge_score_mean double precision,
  latency_ms_p50 double precision,
  latency_ms_p95 double precision,
  latency_ms_p99 double precision,
  pass boolean,
  baseline_id uuid references public.rag_eval_baselines (id),
  git_sha text,
  eval_package_version text,
  run_meta jsonb,
  summary_json jsonb,
  report_storage_path text,
  notes text
);

create index if not exists rag_eval_runs_env_created_idx
  on public.rag_eval_runs (env, created_at desc);

create index if not exists rag_eval_runs_gold_sha_idx
  on public.rag_eval_runs (gold_dataset_sha256);

-- Service role bypasses RLS; enable RLS + policies if exposing to client apps.
alter table public.rag_eval_baselines enable row level security;
alter table public.rag_eval_runs enable row level security;
