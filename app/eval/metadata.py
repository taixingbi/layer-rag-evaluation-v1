"""Run metadata attached to eval summary JSON."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.eval.dataset_version import (
    build_gold_dataset_version,
    build_ingest_manifest_version,
    resolve_ingest_manifest,
)


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
    gold_rows_loaded: int,
    gold_rows_evaluated: int,
    ingest_manifest_path: str | None = None,
    enable_llm_judge: bool = False,
    llm_judge_concurrency: int | None = None,
    llm_judge_model: str | None = None,
    llm_judge_base_url: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
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
        "gold_rows_loaded": gold_rows_loaded,
        "gold_rows_evaluated": gold_rows_evaluated,
    }
    meta.update(build_gold_dataset_version(gold_paths))

    manifest = resolve_ingest_manifest(cli_path=ingest_manifest_path, gold_paths=gold_paths)
    if manifest is not None:
        meta.update(build_ingest_manifest_version(manifest))

    if enable_llm_judge:
        meta["enable_llm_judge"] = True
        meta["llm_judge_concurrency"] = llm_judge_concurrency
        meta["llm_judge_model"] = llm_judge_model
        meta["llm_judge_base_url"] = llm_judge_base_url
    else:
        meta["enable_llm_judge"] = False
    return meta
