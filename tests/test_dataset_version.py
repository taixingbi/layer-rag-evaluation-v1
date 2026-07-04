"""Tests for gold dataset and ingest manifest versioning."""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.baseline import compare_dataset_versions
from app.eval.dataset_version import (
    build_gold_dataset_version,
    build_ingest_manifest_version,
    infer_ingest_env,
    sha256_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLD_FIXTURE = FIXTURES / "gold_single_row.jsonl"


def test_build_gold_dataset_version_stable_hash():
    first = build_gold_dataset_version([GOLD_FIXTURE])
    second = build_gold_dataset_version([GOLD_FIXTURE])
    assert first["gold_dataset_sha256"] == second["gold_dataset_sha256"]
    assert first["gold_dataset_sha256"]
    assert len(first["gold_dataset_files"]) == 1
    assert first["gold_dataset_files"][0]["sha256"] == sha256_file(GOLD_FIXTURE)


def test_infer_ingest_env_from_gold_path():
    assert infer_ingest_env([Path("data_dev/gold_dataset/easy_single_hop.jsonl")]) == "dev"
    assert infer_ingest_env([Path("data_prod/gold_dataset/foo.jsonl")]) == "prod"


def test_build_ingest_manifest_version(tmp_path: Path):
    manifest = tmp_path / "ingest_manifest_latest.json"
    manifest.write_text(
        json.dumps({"env": "dev", "run_id": "run-1", "embed_model": "bge-m3"}),
        encoding="utf-8",
    )
    meta = build_ingest_manifest_version(manifest)
    assert meta["ingest_manifest_path"] == str(manifest)
    assert len(meta["ingest_manifest_sha256"]) == 64
    assert meta["ingest_env"] == "dev"
    assert meta["ingest_run_id"] == "run-1"


def test_compare_dataset_versions_detects_gold_hash_mismatch():
    baseline = {"gold_dataset_sha256": "abc123"}
    current = {"run_meta": {"gold_dataset_sha256": "def456"}}
    msgs = compare_dataset_versions(current, baseline)
    assert any("gold_dataset_sha256" in m for m in msgs)
