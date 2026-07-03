"""LLM-as-judge for RAG answer quality (semantic, not substring heuristics)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.eval.scoring import HEURISTIC_QUALITY_KEYS, quality_score_from_dims
from app.http.inference import async_chat_completions

_JUDGE_SYSTEM = (
    "You are an expert evaluator for retrieval-augmented QA systems. "
    "Score the model answer against the gold reference using semantic correctness, "
    "not exact string matching. Return ONLY valid JSON."
)

_JUDGE_USER_TEMPLATE = """\
Question:
{question}

Gold reference answer:
{gold_answer}

Model answer to evaluate:
{model_answer}

Must-include facts (if any; model should cover these semantically):
{must_contain}

Citation sources returned by the model (may be empty):
{citation_sources}

Expected behavior hint (if any): {expected_behavior}

Score each dimension true/false:
- correct: addresses the question with factually correct content vs gold
- faithful: no unsupported claims; consistent with gold and cited sources
- complete: covers the key facts in gold / must-include list
- precise: focused and not overly vague or hedged when gold is specific
- cited: when factual claims are made, citations align with gold source expectations

Return JSON:
{{"correct": bool, "faithful": bool, "complete": bool, "precise": bool, "cited": bool, "reason": "one short sentence"}}\
"""


def build_judge_messages(
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    must_contain: list[str],
    citation_sources: list[str],
    expected_behavior: str | None,
) -> list[dict[str, str]]:
    must_text = ", ".join(must_contain) if must_contain else "(none)"
    cite_text = ", ".join(citation_sources) if citation_sources else "(none)"
    behavior = (expected_behavior or "").strip() or "(none)"
    user = _JUDGE_USER_TEMPLATE.format(
        question=question.strip(),
        gold_answer=gold_answer.strip() or "(empty)",
        model_answer=model_answer.strip() or "(empty)",
        must_contain=must_text,
        citation_sources=cite_text,
        expected_behavior=behavior,
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_judge_response(content: str) -> dict[str, Any]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge response must be a JSON object, got: {parsed!r}")
    dims: dict[str, bool] = {}
    for key in HEURISTIC_QUALITY_KEYS:
        val = parsed.get(key)
        dims[key] = bool(val) if isinstance(val, bool) else str(val).strip().lower() in {
            "true",
            "1",
            "yes",
        }
    reason = str(parsed.get("reason") or "").strip()
    score = quality_score_from_dims(dims)
    return {
        "llm_judge": dims,
        "llm_judge_score": score,
        "llm_judge_reason": reason,
    }


async def judge_answer_async(
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    must_contain: list[str],
    citation_sources: list[str],
    expected_behavior: str | None,
    base_url: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    messages = build_judge_messages(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
        must_contain=must_contain,
        citation_sources=citation_sources,
        expected_behavior=expected_behavior,
    )
    data = await async_chat_completions(
        messages=messages,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=timeout,
        response_format={"type": "json_object"},
        client=client,
    )
    content = str(data["choices"][0]["message"]["content"]).strip()
    return parse_judge_response(content)
