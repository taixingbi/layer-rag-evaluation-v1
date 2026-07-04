"""Tests for eval run metadata."""

from __future__ import annotations

from pathlib import Path

from app.eval.metadata import build_run_metadata, package_version


def test_build_run_metadata_fields():
    meta = build_run_metadata(
        rag_base_url="http://rag.test",
        collection_base="taixing_knowledge",
        k=5,
        k_max=40,
        gold_paths=[Path("tests/fixtures/gold_single_row.jsonl")],
        recall_ks=[5, 10],
        concurrency=4,
        skip_retrieval_hits=False,
        gold_rows_loaded=10,
        gold_rows_evaluated=5,
    )
    assert meta["gold_rows_loaded"] == 10
    assert meta["gold_rows_evaluated"] == 5
    assert meta["gold_dataset_sha256"]
    assert meta["gold_dataset_files"]
    assert meta["rag_base_url"] == "http://rag.test"
    assert meta["collection_base"] == "taixing_knowledge"
    assert meta["k"] == 5
    assert meta["k_max"] == 40
    assert meta["recall_at_k"] == [5, 10]
    assert meta["eval_package_version"] == package_version()
    assert meta["git_sha"]
    assert meta["timestamp_utc"].endswith("Z")
