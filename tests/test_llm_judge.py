"""Tests for LLM-as-judge (mocked chat API)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.eval.llm_judge import (
    build_judge_messages,
    judge_answer_async,
    parse_judge_response,
)


def test_parse_judge_response_maps_dimensions():
    out = parse_judge_response(
        '{"correct": true, "faithful": false, "complete": true, '
        '"precise": true, "cited": false, "reason": "Missing citation."}'
    )
    assert out["llm_judge"]["correct"] is True
    assert out["llm_judge"]["faithful"] is False
    assert out["llm_judge_score"] == 0.6
    assert out["llm_judge_reason"] == "Missing citation."


def test_build_judge_messages_includes_gold_and_model():
    messages = build_judge_messages(
        question="Where does she work?",
        gold_answer="Saks",
        model_answer="Taixing works at Saks.",
        must_contain=["Saks"],
        citation_sources=["personal_profile"],
        expected_behavior="answer",
    )
    user = messages[1]["content"]
    assert "Saks" in user
    assert "Taixing works at Saks." in user
    assert "personal_profile" in user


@pytest.mark.asyncio
async def test_judge_answer_async_parses_chat_response():
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"correct": true, "faithful": true, "complete": true, '
                        '"precise": true, "cited": true, "reason": "Good match."}'
                    )
                }
            }
        ]
    }

    with patch(
        "app.eval.llm_judge.async_chat_completions",
        new=AsyncMock(return_value=mock_response),
    ) as mock_chat:
        async with httpx.AsyncClient() as client:
            out = await judge_answer_async(
                question="Where does Taixing Bi work?",
                gold_answer="Saks",
                model_answer="She works at Saks.",
                must_contain=["Saks"],
                citation_sources=["personal_profile"],
                expected_behavior="answer",
                base_url="http://llm.test",
                model="test-model",
                api_key=None,
                max_tokens=400,
                timeout=30.0,
                client=client,
            )

    assert out["llm_judge_score"] == 1.0
    assert out["llm_judge_reason"] == "Good match."
    mock_chat.assert_awaited_once()
    assert mock_chat.await_args.kwargs["response_format"] == {"type": "json_object"}


def test_run_eval_row_with_llm_judge_mock():
    from pathlib import Path
    from unittest.mock import patch

    from app.eval.run_eval import _evaluate_all, _load_rows

    fixtures = Path(__file__).parent / "fixtures"
    rows = _load_rows([fixtures / "gold_single_row.jsonl"])

    async def fake_rag_query_async(client, base_url, **kwargs):
        return {
            "answer": "Taixing Bi currently works at Saks.",
            "citations": [{"source": "personal_profile"}],
            "latency_ms": {"total": 1500},
            "retrieval_hits": [],
        }

    judge_out = {
        "llm_judge": {
            "correct": True,
            "faithful": True,
            "complete": True,
            "precise": True,
            "cited": True,
        },
        "llm_judge_score": 1.0,
        "llm_judge_reason": "Semantically correct.",
    }

    with (
        patch("app.eval.run_eval.rag_query_async", side_effect=fake_rag_query_async),
        patch("app.eval.run_eval.judge_answer_async", new=AsyncMock(return_value=judge_out)),
    ):
        results = asyncio.run(
            _evaluate_all(
                rows,
                rag_base_url="http://rag.test",
                collection_base="taixing_knowledge",
                k=5,
                k_max=40,
                concurrency=1,
                limit=None,
                request_retrieval_hits=False,
                recall_ks=[5],
                enable_llm_judge=True,
                llm_judge_base_url="http://llm.test",
                llm_judge_model="test-model",
            )
        )

    assert results[0]["llm_judge_score"] == 1.0
    assert results[0]["heuristic_quality_score"] == 1.0
    assert "llm_judge_error" not in results[0]
