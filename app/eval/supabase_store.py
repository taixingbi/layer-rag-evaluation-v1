"""Supabase persistence for RAG eval runs and baselines (supabase-py v2)."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_supabase_service_key, get_supabase_url
from app.eval.supabase_client import get_supabase_admin_client
from supabase import Client

logger = logging.getLogger(__name__)

RUNS_TABLE = "rag_eval_runs"
BASELINES_TABLE = "rag_eval_baselines"


def supabase_configured(*, env: str | None = None) -> bool:
    from app.core.config import supabase_env_configured

    if env:
        return supabase_env_configured(env)
    for label in ("dev", "prod", "qa"):
        if supabase_env_configured(label):
            return True
    try:
        get_supabase_url(required=True)
        get_supabase_service_key(required=True)
        return True
    except ValueError:
        return False


def _resolve_client(client: Client | None, *, env: str) -> Client:
    return client if client is not None else get_supabase_admin_client(env)


def _first_row(response: Any) -> dict[str, Any] | None:
    data = getattr(response, "data", None)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def _rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    return data if isinstance(data, list) else []


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def baseline_row_to_compare_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Merge baseline columns + ``baseline_json`` for regression compare."""
    out: dict[str, Any] = {}
    nested = row.get("baseline_json")
    if isinstance(nested, dict):
        out.update(nested)
    for key in (
        "rag_calls_failed",
        "mrr_rerank",
        "recall_at_5_rerank",
        "llm_judge_score_mean",
        "heuristic_quality_score_mean",
        "must_contain_pass_rate",
        "gold_dataset_sha256",
        "ingest_manifest_sha256",
        "collection_base",
    ):
        if key in row and row[key] is not None:
            out[key] = row[key]
    out["run_meta"] = {
        k: row.get(k)
        for k in ("gold_dataset_sha256", "ingest_manifest_sha256", "collection_base")
        if row.get(k)
    }
    return out


def summary_to_run_row(
    summary: dict[str, Any],
    *,
    env: str,
    baseline_id: str | None,
    passed: bool,
    report_storage_path: str | None,
    notes: str | None,
) -> dict[str, Any]:
    run_meta = summary.get("run_meta") if isinstance(summary.get("run_meta"), dict) else {}
    row: dict[str, Any] = {
        "env": env,
        "collection_base": run_meta.get("collection_base") or summary.get("collection_base"),
        "gold_dataset_sha256": run_meta.get("gold_dataset_sha256"),
        "ingest_manifest_sha256": run_meta.get("ingest_manifest_sha256"),
        "rows_loaded": _int(run_meta.get("gold_rows_loaded")),
        "rows_evaluated": _int(summary.get("rows") or run_meta.get("gold_rows_evaluated")),
        "rag_calls_failed": _int(summary.get("rag_calls_failed")),
        "mrr_rerank": _float(summary.get("mrr_rerank")),
        "recall_at_5_rerank": _float(summary.get("recall_at_5_rerank")),
        "precision_at_5_rerank": _float(summary.get("precision_at_5_rerank")),
        "ndcg_at_5_rerank": _float(summary.get("ndcg_at_5_rerank")),
        "llm_judge_score_mean": _float(summary.get("llm_judge_score_mean")),
        "latency_ms_p50": _float(summary.get("latency_ms_p50")),
        "latency_ms_p95": _float(summary.get("latency_ms_p95")),
        "latency_ms_p99": _float(summary.get("latency_ms_p99")),
        "pass": passed,
        "baseline_id": baseline_id,
        "git_sha": run_meta.get("git_sha"),
        "eval_package_version": run_meta.get("eval_package_version"),
        "run_meta": run_meta,
        "summary_json": summary,
        "report_storage_path": report_storage_path,
        "notes": notes,
    }
    return {k: v for k, v in row.items() if v is not None}


def summary_to_baseline_row(
    summary: dict[str, Any],
    *,
    env: str,
    name: str,
    notes: str | None,
) -> dict[str, Any]:
    run_meta = summary.get("run_meta") if isinstance(summary.get("run_meta"), dict) else {}
    baseline_json = {
        k: summary.get(k)
        for k in (
            "rag_calls_failed",
            "mrr_rerank",
            "mrr_retrieve",
            "recall_at_5_rerank",
            "recall_at_5_retrieve",
            "recall_at_10_rerank",
            "recall_at_40_rerank",
            "heuristic_quality_score_mean",
            "llm_judge_score_mean",
            "must_contain_pass_rate",
        )
        if summary.get(k) is not None
    }
    if run_meta.get("gold_dataset_sha256"):
        baseline_json["gold_dataset_sha256"] = run_meta["gold_dataset_sha256"]
    if run_meta.get("ingest_manifest_sha256"):
        baseline_json["ingest_manifest_sha256"] = run_meta["ingest_manifest_sha256"]
    if run_meta.get("collection_base"):
        baseline_json["collection_base"] = run_meta["collection_base"]

    row: dict[str, Any] = {
        "env": env,
        "name": name,
        "active": True,
        "collection_base": run_meta.get("collection_base"),
        "gold_dataset_sha256": run_meta.get("gold_dataset_sha256"),
        "ingest_manifest_sha256": run_meta.get("ingest_manifest_sha256"),
        "mrr_rerank": _float(summary.get("mrr_rerank")),
        "recall_at_5_rerank": _float(summary.get("recall_at_5_rerank")),
        "llm_judge_score_mean": _float(summary.get("llm_judge_score_mean")),
        "rag_calls_failed": _int(summary.get("rag_calls_failed")),
        "baseline_json": baseline_json,
        "notes": notes,
    }
    return {k: v for k, v in row.items() if v is not None}


def fetch_active_baseline(
    *,
    env: str,
    name: str | None = None,
    client: Client | None = None,
) -> dict[str, Any] | None:
    sb = _resolve_client(client, env=env)
    query = sb.table(BASELINES_TABLE).select("*").eq("env", env).order("created_at", desc=True).limit(1)
    if name:
        query = query.eq("name", name)
    else:
        query = query.eq("active", True)
    response = query.execute()
    rows = _rows(response)
    return rows[0] if rows else None


def deactivate_baselines(*, env: str, client: Client | None = None) -> None:
    sb = _resolve_client(client, env=env)
    sb.table(BASELINES_TABLE).update({"active": False}).eq("env", env).eq("active", True).execute()


def insert_baseline(
    row: dict[str, Any],
    *,
    env: str,
    client: Client | None = None,
) -> dict[str, Any]:
    sb = _resolve_client(client, env=env)
    response = sb.table(BASELINES_TABLE).insert(row).execute()
    inserted = _first_row(response)
    if inserted is None:
        raise ValueError(f"Unexpected Supabase baseline insert response: {response.data!r}")
    return inserted


def insert_run(row: dict[str, Any], *, env: str, client: Client | None = None) -> dict[str, Any]:
    sb = _resolve_client(client, env=env)
    response = sb.table(RUNS_TABLE).insert(row).execute()
    inserted = _first_row(response)
    if inserted is None:
        raise ValueError(f"Unexpected Supabase run insert response: {response.data!r}")
    return inserted


def pin_baseline(
    summary: dict[str, Any],
    *,
    env: str,
    name: str,
    notes: str | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    row = summary_to_baseline_row(summary, env=env, name=name, notes=notes)
    sb = _resolve_client(client, env=env)
    deactivate_baselines(env=env, client=sb)
    return insert_baseline(row, env=env, client=sb)


def record_run(
    summary: dict[str, Any],
    *,
    env: str,
    baseline_id: str | None,
    passed: bool,
    report_storage_path: str | None = None,
    notes: str | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    row = summary_to_run_row(
        summary,
        env=env,
        baseline_id=baseline_id,
        passed=passed,
        report_storage_path=report_storage_path,
        notes=notes,
    )
    return insert_run(row, env=env, client=client)


def list_recent_runs(
    *,
    env: str,
    limit: int = 20,
    client: Client | None = None,
) -> list[dict[str, Any]]:
    sb = _resolve_client(client, env=env)
    response = (
        sb.table(RUNS_TABLE)
        .select(
            "id,created_at,pass,rows_evaluated,mrr_rerank,recall_at_5_rerank,"
            "llm_judge_score_mean,latency_ms_p95,gold_dataset_sha256"
        )
        .eq("env", env)
        .order("created_at", desc=True)
        .limit(max(1, limit))
        .execute()
    )
    return _rows(response)
