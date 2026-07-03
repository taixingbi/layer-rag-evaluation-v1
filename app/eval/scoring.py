"""Pure scoring helpers for gold JSONL RAG evaluation (no HTTP)."""

from __future__ import annotations

import math
import re
from typing import Any

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.IGNORECASE,
)

HEURISTIC_QUALITY_KEYS = ("correct", "faithful", "complete", "precise", "cited")
LLM_JUDGE_KEYS = HEURISTIC_QUALITY_KEYS


def quality_score_from_dims(dims: dict[str, bool]) -> float:
    vals = [1.0 if bool(dims.get(k)) else 0.0 for k in HEURISTIC_QUALITY_KEYS]
    return (sum(vals) / len(vals)) if vals else 0.0


def parse_recall_ks(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except ValueError:
            continue
        if k > 0 and k not in out:
            out.append(k)
    return sorted(out)


def gold_chunk_id(row: dict[str, Any]) -> str | None:
    rid = str(row.get("id") or "").strip()
    if not rid or not _UUID_RE.match(rid):
        return None
    return rid.lower()


def hits_by_stage(retrieval_hits: Any) -> dict[str, list[str]]:
    if not isinstance(retrieval_hits, list):
        return {}
    by_stage: dict[str, list[tuple[int, str]]] = {}
    for h in retrieval_hits:
        if not isinstance(h, dict):
            continue
        st = str(h.get("stage") or "").strip()
        cid = str(h.get("chunk_id") or "").strip()
        if not st or not cid:
            continue
        try:
            rk = int(h["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        by_stage.setdefault(st, []).append((rk, cid.lower()))
    out: dict[str, list[str]] = {}
    for st, pairs in by_stage.items():
        pairs.sort(key=lambda x: x[0])
        out[st] = [p[1] for p in pairs]
    return out


def rank_of(chunk_ids: list[str], target: str) -> int | None:
    t = target.lower()
    for i, cid in enumerate(chunk_ids):
        if cid.lower() == t:
            return i + 1
    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def precision_at_k(rank: int | None, k: int) -> float:
    if k <= 0:
        return 0.0
    return (1.0 / float(k)) if hit_at_k(rank, k) else 0.0


def dcg_at_k(rank: int | None, k: int) -> float:
    if not hit_at_k(rank, k) or rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def ndcg_at_k(rank: int | None, k: int) -> float:
    if k <= 0:
        return 0.0
    return dcg_at_k(rank, k)  # ideal DCG = 1 for single relevant item at rank 1


def f1_score(precision: float, recall: float) -> float:
    den = precision + recall
    if den <= 0:
        return 0.0
    return 2.0 * precision * recall / den


def retrieval_row_fields(
    row: dict[str, Any],
    data: dict[str, Any],
    *,
    request_retrieval_hits: bool,
    recall_ks: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "retrieval_scored": False,
        "retrieval_eval_skipped": None,
        "gold_chunk_id": None,
        "rank_retrieve": None,
        "rank_rerank": None,
        "rr_retrieve": 0.0,
        "rr_rerank": 0.0,
    }
    if not request_retrieval_hits:
        out["retrieval_eval_skipped"] = "skip_retrieval_hits_flag"
        return out

    gold = gold_chunk_id(row)
    out["gold_chunk_id"] = gold
    if not gold:
        out["retrieval_eval_skipped"] = "no_gold_uuid_id"
        return out

    raw_hits = data.get("retrieval_hits")
    if not isinstance(raw_hits, list):
        out["retrieval_eval_skipped"] = "no_retrieval_hits_in_response"
        return out
    if not raw_hits:
        out["retrieval_eval_skipped"] = "empty_retrieval_hits"
        return out

    by_stage = hits_by_stage(raw_hits)
    retrieve_ids = by_stage.get("retrieve", [])
    rerank_ids = by_stage.get("rerank", [])

    rank_r = rank_of(retrieve_ids, gold)
    rank_rr = rank_of(rerank_ids, gold)
    out["rank_retrieve"] = rank_r
    out["rank_rerank"] = rank_rr
    out["rr_retrieve"] = reciprocal_rank(rank_r)
    out["rr_rerank"] = reciprocal_rank(rank_rr)
    out["retrieval_hit_count"] = len(raw_hits)
    out["retrieval_stages"] = sorted(by_stage.keys())

    for kk in recall_ks:
        hit_r = hit_at_k(rank_r, kk)
        hit_rr = hit_at_k(rank_rr, kk)
        rec_r = 1.0 if hit_r else 0.0
        rec_rr = 1.0 if hit_rr else 0.0
        pre_r = precision_at_k(rank_r, kk)
        pre_rr = precision_at_k(rank_rr, kk)
        ndcg_r = ndcg_at_k(rank_r, kk)
        ndcg_rr = ndcg_at_k(rank_rr, kk)

        out[f"hit_retrieve_at_{kk}"] = hit_r
        out[f"hit_rerank_at_{kk}"] = hit_rr
        out[f"precision_at_{kk}_retrieve"] = pre_r
        out[f"precision_at_{kk}_rerank"] = pre_rr
        out[f"ndcg_at_{kk}_retrieve"] = ndcg_r
        out[f"ndcg_at_{kk}_rerank"] = ndcg_rr
        out[f"f1_at_{kk}_retrieve"] = f1_score(pre_r, rec_r)
        out[f"f1_at_{kk}_rerank"] = f1_score(pre_rr, rec_rr)

    out["retrieval_scored"] = True
    return out


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def must_contain_hits(answer: str, fragments: list[str]) -> tuple[int, int, list[str]]:
    hay = normalize_answer(answer)
    hits = 0
    missing: list[str] = []
    for frag in fragments:
        f = str(frag).strip()
        if not f:
            continue
        needle = normalize_answer(f)
        if needle and needle in hay:
            hits += 1
        else:
            missing.append(f)
    total = len([f for f in fragments if str(f).strip()])
    return hits, total, missing


def citation_sources(citations: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(citations, list):
        return out
    for c in citations:
        if isinstance(c, dict) and c.get("source"):
            out.add(str(c["source"]))
    return out


def gold_source_hit(row: dict[str, Any], cite_sources: set[str]) -> bool | None:
    gold_src = str(row.get("source") or "").strip()
    if not gold_src or gold_src in ("multi", "negative"):
        return None
    return gold_src in cite_sources


def required_sources_hit(row: dict[str, Any], cite_sources: set[str]) -> bool | None:
    req = row.get("required_sources")
    if not isinstance(req, list) or not req:
        return None
    needed = {str(s).strip() for s in req if str(s).strip()}
    if not needed:
        return None
    return needed.issubset(cite_sources)


def heuristic_quality(
    *,
    cite_sources: set[str],
    must_contain_pass: bool,
    gold_source_hit_val: bool | None,
    required_sources_pass: bool | None,
) -> dict[str, Any]:
    """Proxy answer-quality dimensions (not LLM-judge or human labels)."""
    has_citations = len(cite_sources) > 0

    if required_sources_pass is not None:
        cited = bool(required_sources_pass)
    elif gold_source_hit_val is not None:
        cited = bool(gold_source_hit_val)
    else:
        cited = has_citations

    correct = bool(must_contain_pass)
    complete = bool(must_contain_pass)
    faithful = cited
    precise = bool(correct and faithful)

    dims: dict[str, bool] = {
        "correct": correct,
        "faithful": faithful,
        "complete": complete,
        "precise": precise,
        "cited": cited,
    }
    return {
        "heuristic_quality": dims,
        "heuristic_quality_score": quality_score_from_dims(dims),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 100:
        return values[-1]
    idx = (len(values) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    w = idx - lo
    return values[lo] * (1.0 - w) + values[hi] * w


def summarize(
    results: list[dict[str, Any]],
    *,
    recall_ks: list[int],
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = len(results)
    ok = sum(1 for r in results if r.get("ok"))
    must_pass = sum(1 for r in results if r.get("must_contain_pass"))
    must_scored = sum(1 for r in results if (r.get("must_contain_total") or 0) > 0)

    src_vals = [r.get("gold_source_hit") for r in results if r.get("gold_source_hit") is not None]
    src_pass = sum(1 for v in src_vals if v is True)

    req_vals = [r.get("required_sources_pass") for r in results if r.get("required_sources_pass") is not None]
    req_pass = sum(1 for v in req_vals if v is True)

    errors = [r for r in results if r.get("error")]

    out: dict[str, Any] = {
        "rows": n,
        "rag_calls_ok": ok,
        "rag_calls_failed": n - ok,
        "must_contain_pass": must_pass,
        "must_contain_scored_rows": must_scored,
        "gold_source_checked": len(src_vals),
        "gold_source_pass": src_pass,
        "required_sources_checked": len(req_vals),
        "required_sources_pass": req_pass,
        "errors_sample": errors[:5],
    }
    if must_scored > 0:
        out["must_contain_pass_rate"] = must_pass / must_scored
    else:
        out["must_contain_pass_rate"] = 0.0
    latencies = sorted(
        float(r["latency_ms_total"])
        for r in results
        if r.get("ok") and isinstance(r.get("latency_ms_total"), (int, float))
    )
    nl = len(latencies)
    out["latency_scored_rows"] = nl
    if nl > 0:
        out["latency_ms_mean"] = sum(latencies) / nl
        out["latency_ms_min"] = latencies[0]
        out["latency_ms_max"] = latencies[-1]
        out["latency_ms_p50"] = percentile(latencies, 50)
        out["latency_ms_p95"] = percentile(latencies, 95)
        out["latency_ms_p99"] = percentile(latencies, 99)
    else:
        out["latency_ms_mean"] = 0.0
        out["latency_ms_min"] = 0.0
        out["latency_ms_max"] = 0.0
        out["latency_ms_p50"] = 0.0
        out["latency_ms_p95"] = 0.0
        out["latency_ms_p99"] = 0.0

    hq_rows = [r for r in results if r.get("ok") and isinstance(r.get("heuristic_quality"), dict)]
    nq = len(hq_rows)
    out["heuristic_quality_scored_rows"] = nq
    if nq > 0:
        for k in HEURISTIC_QUALITY_KEYS:
            out[f"heuristic_quality_{k}_pass"] = sum(
                1 for r in hq_rows if bool((r.get("heuristic_quality") or {}).get(k))
            )
            out[f"heuristic_quality_{k}_rate"] = out[f"heuristic_quality_{k}_pass"] / nq
        out["heuristic_quality_score_mean"] = (
            sum(float(r.get("heuristic_quality_score") or 0.0) for r in hq_rows) / nq
        )
    else:
        for k in HEURISTIC_QUALITY_KEYS:
            out[f"heuristic_quality_{k}_pass"] = 0
            out[f"heuristic_quality_{k}_rate"] = 0.0
        out["heuristic_quality_score_mean"] = 0.0

    lj_rows = [
        r
        for r in results
        if r.get("ok") and isinstance(r.get("llm_judge"), dict) and not r.get("llm_judge_error")
    ]
    nl = len(lj_rows)
    out["llm_judge_scored_rows"] = nl
    out["llm_judge_failed_rows"] = sum(1 for r in results if r.get("llm_judge_error"))
    if nl > 0:
        for k in LLM_JUDGE_KEYS:
            out[f"llm_judge_{k}_pass"] = sum(
                1 for r in lj_rows if bool((r.get("llm_judge") or {}).get(k))
            )
            out[f"llm_judge_{k}_rate"] = out[f"llm_judge_{k}_pass"] / nl
        out["llm_judge_score_mean"] = (
            sum(float(r.get("llm_judge_score") or 0.0) for r in lj_rows) / nl
        )
    else:
        for k in LLM_JUDGE_KEYS:
            out[f"llm_judge_{k}_pass"] = 0
            out[f"llm_judge_{k}_rate"] = 0.0
        out["llm_judge_score_mean"] = 0.0

    scored = [r for r in results if r.get("ok") and r.get("retrieval_scored")]
    ns = len(scored)
    out["retrieval_scored_rows"] = ns
    if ns > 0:
        out["mean_rr_retrieve"] = sum(float(r.get("rr_retrieve") or 0) for r in scored) / ns
        out["mean_rr_rerank"] = sum(float(r.get("rr_rerank") or 0) for r in scored) / ns
        out["mrr_retrieve"] = out["mean_rr_retrieve"]
        out["mrr_rerank"] = out["mean_rr_rerank"]
        found_r = [r for r in scored if r.get("rank_retrieve") is not None]
        found_rr = [r for r in scored if r.get("rank_rerank") is not None]
        out["retrieval_found_retrieve"] = len(found_r)
        out["retrieval_found_rerank"] = len(found_rr)
        if found_r:
            out["mean_rr_retrieve_when_found"] = (
                sum(1.0 / int(r["rank_retrieve"]) for r in found_r) / len(found_r)
            )
        else:
            out["mean_rr_retrieve_when_found"] = 0.0
        if found_rr:
            out["mean_rr_rerank_when_found"] = (
                sum(1.0 / int(r["rank_rerank"]) for r in found_rr) / len(found_rr)
            )
        else:
            out["mean_rr_rerank_when_found"] = 0.0
        for kk in recall_ks:
            hk = f"hit_retrieve_at_{kk}"
            out[f"recall_at_{kk}_retrieve"] = sum(1 for r in scored if r.get(hk) is True) / ns
            hk2 = f"hit_rerank_at_{kk}"
            out[f"recall_at_{kk}_rerank"] = sum(1 for r in scored if r.get(hk2) is True) / ns
            out[f"precision_at_{kk}_retrieve"] = (
                sum(float(r.get(f"precision_at_{kk}_retrieve") or 0) for r in scored) / ns
            )
            out[f"precision_at_{kk}_rerank"] = (
                sum(float(r.get(f"precision_at_{kk}_rerank") or 0) for r in scored) / ns
            )
            out[f"ndcg_at_{kk}_retrieve"] = (
                sum(float(r.get(f"ndcg_at_{kk}_retrieve") or 0) for r in scored) / ns
            )
            out[f"ndcg_at_{kk}_rerank"] = (
                sum(float(r.get(f"ndcg_at_{kk}_rerank") or 0) for r in scored) / ns
            )
            out[f"f1_at_{kk}_retrieve"] = (
                sum(float(r.get(f"f1_at_{kk}_retrieve") or 0) for r in scored) / ns
            )
            out[f"f1_at_{kk}_rerank"] = (
                sum(float(r.get(f"f1_at_{kk}_rerank") or 0) for r in scored) / ns
            )
    else:
        out["mean_rr_retrieve"] = 0.0
        out["mean_rr_rerank"] = 0.0
        out["mrr_retrieve"] = 0.0
        out["mrr_rerank"] = 0.0
        out["retrieval_found_retrieve"] = 0
        out["retrieval_found_rerank"] = 0
        out["mean_rr_retrieve_when_found"] = 0.0
        out["mean_rr_rerank_when_found"] = 0.0
        for kk in recall_ks:
            out[f"recall_at_{kk}_retrieve"] = 0.0
            out[f"recall_at_{kk}_rerank"] = 0.0
            out[f"precision_at_{kk}_retrieve"] = 0.0
            out[f"precision_at_{kk}_rerank"] = 0.0
            out[f"ndcg_at_{kk}_retrieve"] = 0.0
            out[f"ndcg_at_{kk}_rerank"] = 0.0
            out[f"f1_at_{kk}_retrieve"] = 0.0
            out[f"f1_at_{kk}_rerank"] = 0.0

    if run_meta is not None:
        out["run_meta"] = run_meta
    return out
