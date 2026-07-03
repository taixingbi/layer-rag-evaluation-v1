"""Evaluation configuration from environment (``.env`` at repo root when present)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from app.core.paths import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DEFAULT_K = 5
DEFAULT_K_MAX = 40
DEFAULT_RAG_CONCURRENCY = 40
DEFAULT_RAG_MAX_ATTEMPTS = 3
DEFAULT_RAG_RETRY_BACKOFF_SEC = 1.0
DEFAULT_LLM_JUDGE_CONCURRENCY = 10
DEFAULT_LLM_JUDGE_TIMEOUT = 120.0
DEFAULT_LLM_JUDGE_MAX_TOKENS = 400

DEFAULT_CHAT_MODEL = (
    os.getenv("CHAT_MODEL") or os.getenv("INFERENCE_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
).strip()


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(
            f"{name} is required. Copy `.env.example` to `.env` and set it, "
            f"or export {name} in your shell."
        )
    return value


def get_rag_base_url() -> str:
    """RAG gateway base URL (no ``/v1`` suffix)."""
    return _require("RAG_BASE_URL").rstrip("/")


def get_rag_collection_base() -> str:
    """``collection_base`` sent to ``POST /v1/rag/query``."""
    return _require("RAG_COLLECTION_BASE")


def get_inference_base_url(*, required: bool = False) -> str:
    """Chat API root for optional gold-generator LLM ``must_contain`` extraction."""
    url = (
        os.getenv("INFERENCE_BASE_URL")
        or os.getenv("CHAT_BASE_URL")
        or os.getenv("INFERENCE_URL")
        or ""
    ).strip().rstrip("/")
    if required and not url:
        raise ValueError(
            "INFERENCE_BASE_URL, CHAT_BASE_URL, or INFERENCE_URL is required for LLM must_contain. "
            "Set one in `.env` or pass --chat-base-url."
        )
    return url


def get_llm_judge_base_url(*, required: bool = False) -> str:
    """Chat API root for ``run_eval --enable-llm-judge``."""
    url = (
        os.getenv("LLM_JUDGE_URL")
        or os.getenv("INFERENCE_URL")
        or os.getenv("CHAT_BASE_URL")
        or os.getenv("INFERENCE_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if required and not url:
        raise ValueError(
            "LLM_JUDGE_URL, INFERENCE_URL, or CHAT_BASE_URL is required with --enable-llm-judge. "
            "Set one in `.env` or pass --llm-judge-base-url."
        )
    return url


def get_llm_judge_model() -> str:
    model = (
        os.getenv("LLM_JUDGE_MODEL")
        or os.getenv("CHAT_MODEL")
        or os.getenv("INFERENCE_MODEL")
        or DEFAULT_CHAT_MODEL
    ).strip()
    return model or DEFAULT_CHAT_MODEL


def get_llm_judge_api_key() -> str | None:
    key = (os.getenv("LLM_JUDGE_API_KEY") or os.getenv("CHAT_API_KEY") or "").strip()
    return key or None


def get_llm_judge_max_tokens() -> int:
    raw = (os.getenv("LLM_JUDGE_MAX_TOKENS") or "").strip()
    if not raw:
        return DEFAULT_LLM_JUDGE_MAX_TOKENS
    try:
        return max(64, int(raw))
    except ValueError:
        return DEFAULT_LLM_JUDGE_MAX_TOKENS


def get_llm_judge_timeout() -> float:
    raw = (os.getenv("LLM_JUDGE_TIMEOUT") or "").strip()
    if not raw:
        return DEFAULT_LLM_JUDGE_TIMEOUT
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_LLM_JUDGE_TIMEOUT
