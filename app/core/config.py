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


def get_supabase_url(*, required: bool = False, env: str | None = None) -> str:
    """Project URL. With ``env=dev|prod``, prefers ``SUPABASE_URL_DEV`` / ``SUPABASE_URL_PROD``."""
    suffix = _supabase_env_suffix(env)
    candidates: list[str] = []
    if suffix:
        candidates.append(f"SUPABASE_URL_{suffix}")
    candidates.append("SUPABASE_URL")
    for name in candidates:
        url = (os.getenv(name) or "").strip().rstrip("/")
        if url:
            return url
    if required:
        if suffix:
            raise ValueError(
                f"SUPABASE_URL_{suffix} (or SUPABASE_URL) is required for --supabase-env {env}."
            )
        raise ValueError(
            "SUPABASE_URL is required. Set it in `.env` for --record-supabase / --baseline-supabase."
        )
    return ""


def _supabase_env_suffix(env: str | None) -> str | None:
    if not env:
        return None
    normalized = env.strip().lower()
    mapping = {"dev": "DEV", "prod": "PROD", "qa": "QA"}
    return mapping.get(normalized)


def _supabase_secret_key_names(suffix: str | None) -> list[str]:
    names: list[str] = []
    if suffix:
        names.extend(
            (
                f"SUPABASE_SECRET_KEY_{suffix}",
                f"SUPABASE_SERVICE_ROLE_KEY_{suffix}",
                f"SUPABASE_SERVICE_KEY_{suffix}",
            )
        )
    names.extend(
        (
            "SUPABASE_SECRET_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_SERVICE_KEY",
        )
    )
    return names


def get_supabase_service_key(*, required: bool = False, env: str | None = None) -> str:
    """Elevated Supabase key for server-side eval persistence.

    With ``env=dev|prod``, prefers ``SUPABASE_SECRET_KEY_DEV`` / ``SUPABASE_SECRET_KEY_PROD``
    (and legacy ``SUPABASE_SERVICE_ROLE_KEY_*``), then falls back to generic names.
    """
    suffix = _supabase_env_suffix(env)
    for name in _supabase_secret_key_names(suffix):
        key = (os.getenv(name) or "").strip()
        if key:
            return key
    if required:
        if suffix:
            raise ValueError(
                f"SUPABASE_SECRET_KEY_{suffix} (or SUPABASE_SECRET_KEY) is required for "
                f"--supabase-env {env}. Set a secret key from Dashboard → Settings → API Keys."
            )
        raise ValueError(
            "SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) is required for Supabase eval "
            "persistence. Set a secret key from Dashboard → Settings → API Keys."
        )
    return key


def get_supabase_secret_key(*, required: bool = False, env: str | None = None) -> str:
    """Alias for :func:`get_supabase_service_key` (Supabase v2 naming)."""
    return get_supabase_service_key(required=required, env=env)


def supabase_env_configured(env: str) -> bool:
    """True when URL + secret key are set for the given eval env label."""
    try:
        get_supabase_url(required=True, env=env)
        get_supabase_service_key(required=True, env=env)
        return True
    except ValueError:
        return False
