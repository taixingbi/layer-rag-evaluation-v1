"""Shared async client for ``POST /v1/rag/query`` (batch eval)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.core.config import (
    DEFAULT_K,
    DEFAULT_K_MAX,
    DEFAULT_RAG_MAX_ATTEMPTS,
    DEFAULT_RAG_RETRY_BACKOFF_SEC,
)


def retryable_http_status(code: int) -> bool:
    return code == 408 or code == 429 or code >= 500


def build_rag_query_body(
    *,
    question: str,
    collection_base: str,
    k: int,
    k_max: int,
    include_retrieval_hits: bool = False,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "question": question,
        "collection_base": collection_base,
        "k": k,
        "k_max": k_max,
        "stream": False,
        "expand_on_not_found": False,
        "include_follow_up_questions": False,
    }
    if include_retrieval_hits:
        body["include_retrieval_hits"] = True
    return body


def build_rag_query_headers(*, request_id: str, session_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
        "X-Session-Id": session_id,
    }


async def rag_query_async(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    k: int = DEFAULT_K,
    k_max: int = DEFAULT_K_MAX,
    include_retrieval_hits: bool = False,
    per_request_timeout: float = 120.0,
    max_attempts: int = DEFAULT_RAG_MAX_ATTEMPTS,
    retry_backoff_sec: float = DEFAULT_RAG_RETRY_BACKOFF_SEC,
    log_prefix: str = "rag_query",
) -> dict[str, Any]:
    """POST ``/v1/rag/query`` with correlation headers (not in JSON body)."""
    url = f"{base_url.rstrip('/')}/v1/rag/query"
    body = build_rag_query_body(
        question=question,
        collection_base=collection_base,
        k=k,
        k_max=k_max,
        include_retrieval_hits=include_retrieval_hits,
    )
    headers = build_rag_query_headers(request_id=request_id, session_id=session_id)
    attempts = max(1, max_attempts)

    for attempt in range(attempts):
        try:
            r = await client.post(
                url,
                json=body,
                headers=headers,
                timeout=per_request_timeout,
            )
            if r.status_code >= 400:
                if retryable_http_status(r.status_code) and attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[{log_prefix}] HTTP {r.status_code}, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or "answer" not in data:
                if attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[{log_prefix}] bad JSON body, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise ValueError(f"Unexpected response: {data!r}")
            return data
        except (httpx.RequestError, ValueError) as exc:
            if attempt + 1 >= attempts:
                raise
            delay = retry_backoff_sec * (2**attempt)
            print(
                f"[{log_prefix}] {type(exc).__name__}: {exc!s}, sleep {delay:.1f}s "
                f"(attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("rag_query_async exhausted attempts")
