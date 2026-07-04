"""Gold dataset and ingest manifest fingerprints for reproducible eval runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.paths import INGEST_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_gold_dataset_version(paths: list[Path]) -> dict[str, Any]:
    """Fingerprint gold JSONL files (content hash + combined dataset hash)."""
    files: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted({p.resolve() for p in paths if p.is_file()}, key=str):
        file_hash = sha256_file(path)
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "sha256": file_hash,
                "bytes": stat.st_size,
            }
        )
        combined.update(str(path).encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
    return {
        "gold_dataset_files": files,
        "gold_dataset_sha256": combined.hexdigest() if files else "",
    }


def infer_ingest_env(gold_paths: list[Path]) -> str | None:
    """Infer dev|qa|prod from paths like ``data_dev/gold_dataset/...``."""
    for path in gold_paths:
        for part in path.parts:
            if part.startswith("data_") and part != "data":
                return part.removeprefix("data_")
    return None


def default_ingest_manifest_path(env: str) -> Path:
    return INGEST_ROOT / f"data_{env}" / "data1" / "processed" / "ingest_manifest_latest.json"


def resolve_ingest_manifest(
    *,
    cli_path: str | None,
    gold_paths: list[Path],
) -> Path | None:
    explicit = (cli_path or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    env = infer_ingest_env(gold_paths)
    if not env:
        return None
    candidate = default_ingest_manifest_path(env)
    return candidate if candidate.is_file() else None


def build_ingest_manifest_version(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    meta: dict[str, Any] = {
        "ingest_manifest_path": str(path),
        "ingest_manifest_sha256": sha256_bytes(raw),
        "ingest_manifest_bytes": len(raw),
    }
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return meta
    if isinstance(parsed, dict):
        for key in ("run_id", "env", "collection_name", "embed_model", "git_sha"):
            if key in parsed:
                meta[f"ingest_{key}"] = parsed.get(key)
    return meta
