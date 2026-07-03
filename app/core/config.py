"""Evaluation defaults from environment (optional ``.env`` at repo root)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from app.core.paths import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DEFAULT_RAG_BASE_URL = (
    os.getenv("RAG_BASE_URL") or "http://192.168.86.179:30183"
).strip().rstrip("/")
DEFAULT_COLLECTION_BASE = (
    os.getenv("RAG_COLLECTION_BASE") or "taixing_knowledge"
).strip()

DEFAULT_K = 5
DEFAULT_K_MAX = 40
DEFAULT_RAG_CONCURRENCY = 40
DEFAULT_RAG_MAX_ATTEMPTS = 3
DEFAULT_RAG_RETRY_BACKOFF_SEC = 1.0

INFERENCE_BASE_URL = (
    os.getenv("INFERENCE_BASE_URL")
    or os.getenv("CHAT_BASE_URL")
    or os.getenv("INFERENCE_URL")
    or "http://192.168.86.179:30180"
).strip().rstrip("/")
DEFAULT_CHAT_MODEL = (
    os.getenv("CHAT_MODEL")
    or os.getenv("INFERENCE_MODEL")
    or "Qwen/Qwen2.5-7B-Instruct"
).strip()
