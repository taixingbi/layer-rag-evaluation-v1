"""Compare eval summaries against a pinned baseline (regression gate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Metrics where higher is better (rate or score in [0, 1] typically).
HIGHER_IS_BETTER = (
    "mrr_retrieve",
    "mrr_rerank",
    "mean_rr_retrieve_when_found",
    "mean_rr_rerank_when_found",
    "heuristic_quality_score_mean",
    "llm_judge_score_mean",
    "must_contain_pass_rate",
    "recall_at_5_retrieve",
    "recall_at_5_rerank",
    "recall_at_10_retrieve",
    "recall_at_10_rerank",
    "recall_at_40_retrieve",
    "recall_at_40_rerank",
)


def load_baseline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline must be a JSON object: {path}")
    return data


def compare_summaries(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerance: float,
) -> list[str]:
    """Return human-readable regression messages (empty if within tolerance)."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    failures: list[str] = []
    for key in HIGHER_IS_BETTER:
        if key not in baseline:
            continue
        base_val = baseline.get(key)
        cur_val = current.get(key)
        if not isinstance(base_val, (int, float)) or not isinstance(cur_val, (int, float)):
            continue
        if cur_val < float(base_val) - tolerance:
            failures.append(
                f"{key}: {cur_val:.4f} < baseline {float(base_val):.4f} - tolerance {tolerance}"
            )

    base_fail = baseline.get("rag_calls_failed")
    cur_fail = current.get("rag_calls_failed")
    if isinstance(base_fail, int) and isinstance(cur_fail, int) and cur_fail > base_fail:
        failures.append(f"rag_calls_failed: {cur_fail} > baseline {base_fail}")

    return failures


def compare_dataset_versions(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    """Fail when pinned baseline specifies dataset fingerprints that differ."""
    failures: list[str] = []
    cur_meta = current.get("run_meta") if isinstance(current.get("run_meta"), dict) else {}
    base_meta = baseline.get("run_meta") if isinstance(baseline.get("run_meta"), dict) else {}

    for key in ("gold_dataset_sha256", "ingest_manifest_sha256"):
        expected = baseline.get(key) or base_meta.get(key)
        if not expected:
            continue
        actual = cur_meta.get(key) or current.get(key)
        if actual != expected:
            failures.append(f"{key}: {actual!r} != baseline {expected!r}")

    base_collection = baseline.get("collection_base") or base_meta.get("collection_base")
    cur_collection = cur_meta.get("collection_base") or current.get("collection_base")
    if base_collection and cur_collection and cur_collection != base_collection:
        failures.append(f"collection_base: {cur_collection!r} != baseline {base_collection!r}")

    return failures
