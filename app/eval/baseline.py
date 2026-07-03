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
