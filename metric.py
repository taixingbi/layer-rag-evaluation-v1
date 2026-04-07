"""Add evaluation metrics to each row of a gold / inference JSON dataset.

Primary path: **LLM-as-judge** via OpenAI-compatible ``POST .../v1/chat/completions``
(``LLM_JUDGE_URL`` or ``INFERENCE_URL`` in ``.env``).

Each row gets ``metrics``. With a judge URL configured, **all five** scores are
produced by **one** LLM-as-judge completion (no per-metric formulas):

- ``faithfulness_grounding`` — judge: grounding vs gold and excerpts.
- ``answer_correctness`` — judge: alignment with gold.
- ``context_relevance`` — judge: how well excerpts support the answer (``0.0`` if none).
- ``answer_relevance`` — judge: whether the answer addresses the question.
- ``hallucination_rate_proxy`` — judge: contradiction/fabrication vs gold (lower is better).

When a judge URL is set, there is no blending with string/embed scores. On failure,
see ``_metrics_llm_unavailable``.

``--heuristic-only`` or a missing judge URL uses local heuristics instead (then
``context_relevance`` is always ``null``).

Example judge API (one-shot)::

    curl http://HOST:PORT/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "..."}], "max_tokens": 400}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_CITE_RE = re.compile(r"\[\d+\]")
_NOT_FOUND = "NOT_FOUND"
_METRIC_KEYS = (
    "faithfulness_grounding",
    "answer_correctness",
    "context_relevance",
    "answer_relevance",
    "hallucination_rate_proxy",
)

_EVA_ROOT = Path(__file__).resolve().parent
_DEFAULT_IN = _EVA_ROOT / "result" / "dataset-gold-test-1.0.0.json"
_DEFAULT_OUT = _EVA_ROOT / "result" / "dataset-gold-test-eva-1.0.0.json"
_DEFAULT_JUDGE_MAX_ATTEMPTS = 3
_DEFAULT_JUDGE_RETRY_BACKOFF_SEC = 1.0
_MAX_CONCURRENCY_DEFAULT = 20


def _retryable_judge_http(code: int) -> bool:
    return code == 408 or code == 429 or code >= 500


def _strip_citations(s: str) -> str:
    return _CITE_RE.sub("", s).strip()


def _normalize(s: str) -> str:
    t = (s or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _prediction_text(row: dict) -> str:
    """Prefer ``inference-output``; fall back to ``answer`` (RAG file-api shape)."""
    raw = (row.get("inference-output") or row.get("answer") or "").strip()
    return raw


def _string_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _question_for_metrics(row: dict) -> str:
    """Gold rows use ``input``; API-shaped rows may only have ``question``."""
    return (row.get("input") or row.get("question") or "").strip()


def _citation_excerpts_block(row: dict, *, max_chars: int = 6000) -> str:
    cites = row.get("citations")
    if not isinstance(cites, list) or not cites:
        return "(none)"
    parts: list[str] = []
    n = 0
    for i, c in enumerate(cites, start=1):
        if not isinstance(c, dict):
            continue
        ex = (c.get("excerpt") or c.get("source") or "").strip()
        if not ex:
            continue
        line = f"[{i}] {ex}"
        if n + len(line) > max_chars:
            parts.append("… (truncated)")
            break
        parts.append(line)
        n += len(line) + 1
    return "\n".join(parts) if parts else "(none)"


def _embedding_map_for_rows(rows: list[dict]) -> dict[str, list[float]] | None:
    """Single batched ``embed_texts`` over all unique strings needed for metrics."""
    texts: list[str] = []
    for row in rows:
        question = _question_for_metrics(row)
        gold = (row.get("output") or "").strip()
        raw_pred = _prediction_text(row)
        pred_no_cite = _strip_citations(raw_pred)
        abstention = (
            pred_no_cite.upper() == _NOT_FOUND.upper() or raw_pred.strip() == _NOT_FOUND
        )
        pred_for_correctness = raw_pred if abstention else pred_no_cite
        if question:
            texts.append(question)
        if gold:
            texts.append(gold)
        if pred_for_correctness:
            texts.append(pred_for_correctness)
    unique = list(dict.fromkeys(texts))
    if not unique:
        return {}
    try:
        from app.embed import embed_texts
    except ImportError:
        print("Warning: app.embed unavailable; cosine metrics omitted.", file=sys.stderr)
        return None
    try:
        vecs = embed_texts(
            unique,
            request_id="eva-metrics-batch",
            session_id="eva-metrics-session",
        )
    except Exception as e:
        print(f"Warning: reference vector batch failed: {e}", file=sys.stderr)
        return None
    return {t: vecs[i] for i, t in enumerate(unique)}


def _metrics_for_row(
    row: dict,
    *,
    emb_map: dict[str, list[float]] | None,
) -> dict:
    question = _question_for_metrics(row)
    gold = (row.get("output") or "").strip()
    raw_pred = _prediction_text(row)
    pred_no_cite = _strip_citations(raw_pred)

    abstention = pred_no_cite.upper() == _NOT_FOUND.upper() or raw_pred.strip() == _NOT_FOUND
    has_cites = bool(_CITE_RE.search(raw_pred))

    norm_gold = _normalize(gold)
    norm_pred = _normalize(pred_no_cite)

    sim_pred_gold = _string_similarity(norm_pred, norm_gold)
    exact = norm_pred == norm_gold

    pred_for_correctness = raw_pred if abstention else pred_no_cite

    embed_pred_gold: float | None = None
    embed_q_pred: float | None = None
    if emb_map is not None:
        if pred_for_correctness and gold and pred_for_correctness in emb_map and gold in emb_map:
            embed_pred_gold = _cosine(emb_map[pred_for_correctness], emb_map[gold])
        if (
            not abstention
            and question
            and pred_no_cite
            and question in emb_map
            and pred_no_cite in emb_map
        ):
            embed_q_pred = _cosine(emb_map[question], emb_map[pred_no_cite])

    if abstention:
        faith_score = 1.0
    elif has_cites:
        faith_score = 0.75 + 0.25 * sim_pred_gold
    else:
        faith_score = 0.55 + 0.45 * sim_pred_gold

    semantic = embed_pred_gold if embed_pred_gold is not None else sim_pred_gold
    if exact:
        correctness = 1.0
    elif embed_pred_gold is not None:
        correctness = embed_pred_gold
    else:
        correctness = sim_pred_gold

    if abstention and norm_gold:
        hallucination = 0.15
    elif not norm_gold:
        hallucination = 0.0
    else:
        hallucination = max(0.0, min(1.0, 1.0 - semantic))

    q_pred_sim = _string_similarity(_normalize(question), norm_pred) if question else 0.0
    answer_rel = embed_q_pred if embed_q_pred is not None else q_pred_sim

    return {
        "faithfulness_grounding": round(faith_score, 4),
        "answer_correctness": round(correctness, 4),
        "context_relevance": None,
        "answer_relevance": round(answer_rel, 4),
        "hallucination_rate_proxy": round(hallucination, 4),
    }


def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _metrics_llm_unavailable(*, reason: str = "unavailable") -> dict[str, Any]:
    """Placeholder when LLM judge cannot produce scores (no mixing with heuristics)."""
    _ = reason
    return {
        "faithfulness_grounding": 0.0,
        "answer_correctness": 0.0,
        "context_relevance": 0.0,
        "answer_relevance": 0.0,
        "hallucination_rate_proxy": 1.0,
    }


def _finalize_judge_metrics(d: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize LLM JSON: all five keys must be numeric 0--1 (coerce missing context to 0)."""
    out: dict[str, Any] = {}
    for k in _METRIC_KEYS:
        v = d.get(k)
        if v is None or v == "null":
            if k == "context_relevance":
                out[k] = 0.0
                continue
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            if k == "context_relevance":
                out[k] = 0.0
                continue
            return None
        out[k] = round(max(0.0, min(1.0, f)), 4)
    return out


async def _llm_judge_row_async(
    row: dict,
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    max_tokens: int,
    max_attempts: int,
    retry_backoff_sec: float,
) -> dict[str, Any] | None:
    question = _question_for_metrics(row)
    gold = (row.get("output") or "").strip()
    pred = _prediction_text(row)
    if not pred:
        return None

    excerpts = _citation_excerpts_block(row)
    user_msg = f"""You are a strict evaluation judge for RAG answers.

You must assign every score yourself in ONE JSON object (no markdown, no extra text).
Include exactly these five keys, each a number from 0 to 1 inclusive:
"faithfulness_grounding", "answer_correctness", "context_relevance", "answer_relevance", "hallucination_rate_proxy".
Do not omit any key. Do not use null; use 0.0 when a dimension does not apply (e.g. no excerpts → context_relevance 0.0).

Rubric (your judgment only):
- faithfulness_grounding: Answer supported by gold + excerpts? Penalize unsupported claims.
- answer_correctness: Semantic match vs gold reference.
- context_relevance: Do excerpts justify the answer? 0.0 if excerpts are "(none)" or irrelevant.
- answer_relevance: Does the answer address the question?
- hallucination_rate_proxy: 0 = no serious hallucination vs gold; 1 = severe contradiction or fabrication.

Question:
{question}

Gold reference:
{gold}

Model answer:
{pred}

Citation excerpts:
{excerpts}
"""

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    req_json = {
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            r = await client.post(
                url,
                json=req_json,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code >= 400:
                if _retryable_judge_http(r.status_code) and attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[metric] judge HTTP {r.status_code}, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                r.raise_for_status()
            payload = r.json()
            choices = payload.get("choices")
            if not choices or not isinstance(choices, list):
                if attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[metric] judge bad payload, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                return None
            msg = (choices[0] or {}).get("message") or {}
            content = (msg.get("content") or "").strip()
            parsed = _parse_judge_json(content)
            if not isinstance(parsed, dict):
                if attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[metric] judge non-JSON reply, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                return None
            m = _finalize_judge_metrics(parsed)
            if m is None:
                if attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[metric] judge invalid scores, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                return None
            return m
        except httpx.RequestError as e:
            if attempt + 1 >= attempts:
                print(f"Warning: LLM judge failed for row: {e}", file=sys.stderr)
                return None
            delay = retry_backoff_sec * (2**attempt)
            print(
                f"[metric] judge {type(e).__name__}: {e!s}, sleep {delay:.1f}s "
                f"(attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            await asyncio.sleep(delay)
        except (httpx.HTTPStatusError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            if attempt + 1 >= attempts:
                print(f"Warning: LLM judge failed for row: {e}", file=sys.stderr)
                return None
            delay = retry_backoff_sec * (2**attempt)
            print(
                f"[metric] judge {type(e).__name__}: {e!s}, sleep {delay:.1f}s "
                f"(attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            await asyncio.sleep(delay)
    return None


async def _process_judge_row(
    idx: int,
    row: dict,
    *,
    n: int,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    max_tokens: int,
    max_attempts: int,
    retry_backoff_sec: float,
) -> None:
    pred = _prediction_text(row)
    if not pred:
        row["metrics"] = _metrics_llm_unavailable(reason="empty_prediction")
        return
    async with sem:
        print(f"[metric] judge [{idx + 1}/{n}] ...", flush=True)
        m = await _llm_judge_row_async(
            row,
            client=client,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            retry_backoff_sec=retry_backoff_sec,
        )
        if m is None:
            row["metrics"] = _metrics_llm_unavailable(reason="parse_or_http")
        else:
            row["metrics"] = m


async def _attach_metrics_llm_only_async(
    rows: list[dict],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    max_attempts: int,
    retry_backoff_sec: float,
    max_concurrency: int,
) -> None:
    n = len(rows)
    cap = max(1, min(max_concurrency, n))
    sem = asyncio.Semaphore(cap)
    limits = httpx.Limits(max_connections=cap + 5, max_keepalive_connections=cap + 5)
    print(
        f"[metric] LLM judge (sole scorer) → {base_url} model={model} concurrency={cap}",
        flush=True,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(
            *(
                _process_judge_row(
                    idx,
                    row,
                    n=n,
                    client=client,
                    sem=sem,
                    base_url=base_url,
                    model=model,
                    max_tokens=max_tokens,
                    max_attempts=max_attempts,
                    retry_backoff_sec=retry_backoff_sec,
                )
                for idx, row in enumerate(rows)
            )
        )


def _attach_metrics_llm_only(
    rows: list[dict],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    max_attempts: int,
    retry_backoff_sec: float,
    max_concurrency: int,
) -> None:
    asyncio.run(
        _attach_metrics_llm_only_async(
            rows,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            max_attempts=max_attempts,
            retry_backoff_sec=retry_backoff_sec,
            max_concurrency=max_concurrency,
        )
    )


def _attach_metrics_heuristic(rows: list[dict]) -> None:
    emb_map = _embedding_map_for_rows(rows)
    if emb_map is not None and len(emb_map) > 0:
        print(
            f"Batched reference vectors: {len(emb_map)} unique strings → 1 API call.",
            flush=True,
        )
    for row in rows:
        row["metrics"] = _metrics_for_row(row, emb_map=emb_map)


def attach_metrics(
    rows: list[dict],
    *,
    heuristic_only: bool = False,
    judge_max_attempts: int | None = None,
    judge_retry_backoff_sec: float | None = None,
    judge_max_concurrency: int | None = None,
) -> None:
    """Add ``metrics`` to each row in place. LLM judge is the sole scorer when URL is set."""
    load_dotenv(_EVA_ROOT / ".env")
    if heuristic_only:
        _attach_metrics_heuristic(rows)
        return

    base = (os.getenv("LLM_JUDGE_URL") or os.getenv("INFERENCE_URL") or "").strip().rstrip("/")
    if not base:
        print(
            "Warning: LLM_JUDGE_URL and INFERENCE_URL unset; using heuristic metrics.",
            file=sys.stderr,
        )
        _attach_metrics_heuristic(rows)
        return

    model = (
        os.getenv("LLM_JUDGE_MODEL")
        or os.getenv("INFERENCE_MODEL")
        or "Qwen/Qwen2.5-7B-Instruct"
    )
    timeout = float(os.getenv("LLM_JUDGE_TIMEOUT", "120"))
    max_tokens = int(os.getenv("LLM_JUDGE_MAX_TOKENS", "400"))
    j_attempts = (
        judge_max_attempts
        if judge_max_attempts is not None
        else int(os.getenv("LLM_JUDGE_MAX_ATTEMPTS", str(_DEFAULT_JUDGE_MAX_ATTEMPTS)))
    )
    j_backoff = (
        judge_retry_backoff_sec
        if judge_retry_backoff_sec is not None
        else float(os.getenv("LLM_JUDGE_RETRY_BACKOFF", str(_DEFAULT_JUDGE_RETRY_BACKOFF_SEC)))
    )
    j_conc = (
        judge_max_concurrency
        if judge_max_concurrency is not None
        else int(os.getenv("LLM_JUDGE_MAX_CONCURRENCY", str(_MAX_CONCURRENCY_DEFAULT)))
    )
    _attach_metrics_llm_only(
        rows,
        base_url=base,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_attempts=max(1, j_attempts),
        retry_backoff_sec=max(0.0, j_backoff),
        max_concurrency=max(1, j_conc),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Add metrics dict to each dataset row (LLM judge or heuristic).")
    p.add_argument(
        "-i",
        "--in",
        dest="in_path",
        type=Path,
        default=_DEFAULT_IN,
        help="Input JSON (e.g. eva/result/dataset-gold-test-1.0.0.json)",
    )
    p.add_argument(
        "-o",
        "--out",
        dest="out_path",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output JSON (e.g. dataset-gold-test-eva-1.0.0.json)",
    )
    p.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip LLM judge; use string/embed heuristics only",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        metavar="N",
        help="LLM judge retries+1 (default: env LLM_JUDGE_MAX_ATTEMPTS or 3)",
    )
    p.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        metavar="SEC",
        help="LLM judge retry base delay seconds (default: env LLM_JUDGE_RETRY_BACKOFF or 1.0)",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help=f"Max concurrent judge HTTP requests (default: env LLM_JUDGE_MAX_CONCURRENCY or {_MAX_CONCURRENCY_DEFAULT})",
    )
    args = p.parse_args(argv)

    in_path = args.in_path
    out_path = args.out_path
    if not in_path.is_file():
        print(f"Missing input: {in_path}", file=sys.stderr)
        return 1

    with open(in_path, encoding="utf-8") as f:
        rows: list[dict] = json.load(f)
    if not isinstance(rows, list):
        print("Expected JSON array", file=sys.stderr)
        return 1

    attach_metrics(
        rows,
        heuristic_only=args.heuristic_only,
        judge_max_attempts=args.max_attempts,
        judge_retry_backoff_sec=args.retry_backoff,
        judge_max_concurrency=args.max_concurrency,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(out_path)
    print(f"Wrote {len(rows)} rows with metrics to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
