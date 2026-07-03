"""Repository and sibling data paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INGEST_ROOT = REPO_ROOT.parent / "layer-rag-ingest-v1"
