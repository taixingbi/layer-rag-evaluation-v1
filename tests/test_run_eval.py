"""Integration-style tests for run_eval (mocked RAG, no live gateway)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from app.eval.run_eval import _evaluate_all, _load_rows
from app.http.rag import build_rag_query_body, build_rag_query_headers, rag_query_async

FIXTURES = Path(__file__).parent / "fixtures"
GOLD_ROW_ID = "08d53b51-7e9f-58db-8863-fc437b7100fe"


def test_build_rag_query_headers_not_in_body():
    headers = build_rag_query_headers(request_id="req-1", session_id="ses-1")
    body = build_rag_query_body(
        question="Where does Taixing Bi work?",
        collection_base="taixing_knowledge",
        k=5,
        k_max=40,
        include_retrieval_hits=True,
    )
    assert headers["X-Request-Id"] == "req-1"
    assert headers["X-Session-Id"] == "ses-1"
    assert "X-Request-Id" not in body
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_rag_query_async_posts_headers_and_body():
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answer": "Saks",
                "citations": [{"source": "personal_profile"}],
                "latency_ms": {"total": 1200},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        data = await rag_query_async(
            client,
            "http://rag.test",
            question="Where does Taixing Bi work?",
            collection_base="taixing_knowledge",
            request_id="req-abc",
            session_id="ses-xyz",
            k=5,
            k_max=40,
            include_retrieval_hits=True,
            max_attempts=1,
        )

    assert data["answer"] == "Saks"
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers.get("x-request-id") == "req-abc"
    assert headers.get("x-session-id") == "ses-xyz"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["question"] == "Where does Taixing Bi work?"
    assert body["include_retrieval_hits"] is True


def test_evaluate_one_row_mock_rag():
    rows = _load_rows([FIXTURES / "gold_single_row.jsonl"])
    assert len(rows) == 1

    async def fake_rag_query_async(client, base_url, **kwargs):
        assert kwargs["request_id"].startswith("eva-")
        assert kwargs["session_id"].startswith("eva-ses-")
        assert kwargs["include_retrieval_hits"] is True
        return {
            "answer": "Taixing Bi currently works at Saks.",
            "citations": [{"source": "personal_profile"}],
            "latency_ms": {"total": 1500},
            "retrieval_hits": [
                {
                    "stage": "retrieve",
                    "rank": 1,
                    "chunk_id": GOLD_ROW_ID,
                },
                {
                    "stage": "rerank",
                    "rank": 1,
                    "chunk_id": GOLD_ROW_ID,
                },
            ],
        }

    with patch("app.eval.run_eval.rag_query_async", side_effect=fake_rag_query_async):
        results = asyncio.run(
            _evaluate_all(
                rows,
                rag_base_url="http://rag.test",
                collection_base="taixing_knowledge",
                k=5,
                k_max=40,
                concurrency=1,
                limit=None,
                request_retrieval_hits=True,
                recall_ks=[5],
            )
        )

    assert len(results) == 1
    row = results[0]
    assert row["ok"] is True
    assert row["must_contain_pass"] is True
    assert row["rank_retrieve"] == 1
    assert row["rank_rerank"] == 1
    assert row["hit_retrieve_at_5"] is True
    assert row["gold_source_hit"] is True
