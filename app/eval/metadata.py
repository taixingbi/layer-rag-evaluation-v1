"""Run metadata attached to eval summary JSON."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def package_version() -> str:
    try:
        return version("layer-rag-evaluation")
    except PackageNotFoundError:
        return "0.0.0"


def build_run_metadata(
    *,
    rag_base_url: str,
    collection_base: str,
    k: int,
    k_max: int,
    gold_paths: list[Path],
    recall_ks: list[int],
    concurrency: int,
    skip_retrieval_hits: bool,
) -> dict[str, Any]:
    return {
        "eval_package_version": package_version(),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rag_base_url": rag_base_url,
        "collection_base": collection_base,
        "k": k,
        "k_max": k_max,
        "concurrency": concurrency,
        "recall_at_k": recall_ks,
        "skip_retrieval_hits": skip_retrieval_hits,
        "gold_paths": [str(p) for p in gold_paths],
    }
