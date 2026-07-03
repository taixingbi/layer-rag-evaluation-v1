"""Gold dataset evaluation against the RAG ``/v1/rag/query`` API.

Reads JSONL rows (see ``docs/eval.md``), posts each ``question``, then scores retrieval,
``must_contain``, citations, heuristic proxies, and optional LLM-as-judge answer quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from app.core.config import (
    DEFAULT_K,
    DEFAULT_K_MAX,
    DEFAULT_LLM_JUDGE_CONCURRENCY,
    DEFAULT_RAG_CONCURRENCY,
    get_llm_judge_api_key,
    get_llm_judge_base_url,
    get_llm_judge_max_tokens,
    get_llm_judge_model,
    get_llm_judge_timeout,
    get_rag_base_url,
    get_rag_collection_base,
)
from app.eval.baseline import compare_summaries, load_baseline
from app.eval.llm_judge import judge_answer_async
from app.eval.metadata import build_run_metadata
from app.eval.scoring import (
    citation_sources,
    gold_source_hit,
    heuristic_quality,
    must_contain_hits,
    parse_recall_ks,
    required_sources_hit,
    retrieval_row_fields,
    summarize,
)
from app.http.rag import rag_query_async

logger = logging.getLogger(__name__)


def _iter_jsonl_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for child in sorted(p.glob("*.jsonl")):
                if child.is_file():
                    out.append(child)
        else:
            logger.warning("Gold path not found, skipping: %s", raw)
    return out


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skip bad JSON %s:%s: %s", path, line_no, exc)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _resolve_rag_base_url(cli_value: str | None) -> str:
    explicit = (cli_value or "").strip().rstrip("/")
    if explicit:
        return explicit
    return get_rag_base_url()


def _resolve_collection_base(cli_value: str | None) -> str:
    explicit = (cli_value or "").strip()
    if explicit:
        return explicit
    return get_rag_collection_base()


async def _evaluate_all(
    rows: list[dict[str, Any]],
    *,
    rag_base_url: str,
    collection_base: str,
    k: int,
    k_max: int,
    concurrency: int,
    limit: int | None,
    request_retrieval_hits: bool,
    recall_ks: list[int],
    enable_llm_judge: bool = False,
    llm_judge_base_url: str = "",
    llm_judge_model: str = "",
    llm_judge_api_key: str | None = None,
    llm_judge_concurrency: int = DEFAULT_LLM_JUDGE_CONCURRENCY,
    llm_judge_max_tokens: int = 400,
    llm_judge_timeout: float = 120.0,
) -> list[dict[str, Any]]:
    import httpx

    sem = asyncio.Semaphore(max(1, concurrency))
    judge_sem = asyncio.Semaphore(max(1, llm_judge_concurrency))
    work = rows if limit is None else rows[: max(0, limit)]

    async with httpx.AsyncClient() as client:

        async def _one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
            question = str(row.get("question") or "").strip()
            out: dict[str, Any] = {
                "index": idx,
                "env": row.get("env"),
                "id": row.get("id"),
                "eval_bucket": row.get("eval_bucket"),
                "case_type": row.get("case_type"),
                "expected_behavior": row.get("expected_behavior"),
                "question": question,
                "ok": False,
                "http_status": None,
                "error": None,
                "must_contain_hits": 0,
                "must_contain_total": 0,
                "must_contain_missing": [],
                "must_contain_pass": False,
                "gold_source_hit": None,
                "required_sources_pass": None,
                "latency_ms_total": None,
            }
            if not question:
                out["error"] = "empty_question"
                return out

            request_id = f"eva-{uuid.uuid4().hex[:12]}"
            session_id = f"eva-ses-{uuid.uuid4().hex[:10]}"
            try:
                async with sem:
                    data = await rag_query_async(
                        client,
                        rag_base_url,
                        question=question,
                        collection_base=collection_base,
                        request_id=request_id,
                        session_id=session_id,
                        k=k,
                        k_max=k_max,
                        include_retrieval_hits=request_retrieval_hits,
                        max_attempts=1,
                        log_prefix="run_eval",
                    )
            except Exception as exc:
                out["error"] = str(exc)
                return out

            out["ok"] = True
            answer = str(data.get("answer") or "")
            cite_sources = citation_sources(data.get("citations"))

            must_list: list[str] = []
            raw_must = row.get("must_contain")
            if isinstance(raw_must, list):
                must_list = [str(x) for x in raw_must]

            hits, total, missing = must_contain_hits(answer, must_list)
            out["must_contain_hits"] = hits
            out["must_contain_total"] = total
            out["must_contain_missing"] = missing
            out["must_contain_pass"] = total == 0 or (hits == total and total > 0)

            gs_hit = gold_source_hit(row, cite_sources)
            rs_pass = required_sources_hit(row, cite_sources)
            out["gold_source_hit"] = gs_hit
            out["required_sources_pass"] = rs_pass
            out.update(
                heuristic_quality(
                    cite_sources=cite_sources,
                    must_contain_pass=bool(out["must_contain_pass"]),
                    gold_source_hit_val=gs_hit,
                    required_sources_pass=rs_pass,
                )
            )

            lat = data.get("latency_ms")
            if isinstance(lat, dict) and lat.get("total") is not None:
                out["latency_ms_total"] = lat.get("total")

            out["rag_answer_preview"] = answer[:400]
            out["citation_sources"] = sorted(cite_sources)
            out.update(
                retrieval_row_fields(
                    row,
                    data,
                    request_retrieval_hits=request_retrieval_hits,
                    recall_ks=recall_ks,
                )
            )

            if enable_llm_judge:
                gold_answer = str(row.get("answer") or "")
                expected_behavior = row.get("expected_behavior")
                if expected_behavior is not None:
                    expected_behavior = str(expected_behavior)
                try:
                    async with judge_sem:
                        out.update(
                            await judge_answer_async(
                                question=question,
                                gold_answer=gold_answer,
                                model_answer=answer,
                                must_contain=must_list,
                                citation_sources=sorted(cite_sources),
                                expected_behavior=expected_behavior,
                                base_url=llm_judge_base_url,
                                model=llm_judge_model,
                                api_key=llm_judge_api_key,
                                max_tokens=llm_judge_max_tokens,
                                timeout=llm_judge_timeout,
                                client=client,
                            )
                        )
                except Exception as exc:
                    out["llm_judge_error"] = str(exc)

            return out

        tasks = [_one(i, row) for i, row in enumerate(work)]
        return await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate gold JSONL against RAG /v1/rag/query.")
    p.add_argument(
        "--gold",
        nargs="+",
        required=True,
        help="Gold JSONL file(s) or directories containing *.jsonl.",
    )
    p.add_argument(
        "--rag-base-url",
        default="",
        help="RAG gateway base URL (no /v1 suffix). Default: RAG_BASE_URL from .env",
    )
    p.add_argument(
        "--collection-base",
        default="",
        help="collection_base for the API. Default: RAG_COLLECTION_BASE from .env",
    )
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_RAG_CONCURRENCY,
        help="Max concurrent async RAG requests (default: %(default)s).",
    )
    p.add_argument("--limit", type=int, default=0, help="Max rows to evaluate (0 = all).")
    p.add_argument(
        "--skip-retrieval-hits",
        action="store_true",
        help="Do not send include_retrieval_hits; skip retrieval rank / Recall@k metrics.",
    )
    p.add_argument(
        "--recall-at-k",
        default="5,10,40",
        help="Comma-separated k values for Recall@k (default: %(default)s).",
    )
    p.add_argument("--report-json", default="", help="Write per-row results JSON array.")
    p.add_argument("--summary-json", default="", help="Write summary object to this path.")
    p.add_argument(
        "--baseline-json",
        default="",
        help="Pinned summary JSON; fail if key metrics drop below baseline by --baseline-tolerance.",
    )
    p.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.05,
        help="Max allowed drop vs --baseline-json for higher-is-better metrics (default: %(default)s).",
    )
    p.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Score answers with LLM-as-judge (same run; requires chat API URL).",
    )
    p.add_argument(
        "--llm-judge-base-url",
        default="",
        help="Chat API base for judge. Default: LLM_JUDGE_URL / INFERENCE_URL / CHAT_BASE_URL from .env",
    )
    p.add_argument(
        "--llm-judge-model",
        default="",
        help="Judge model name. Default: LLM_JUDGE_MODEL / CHAT_MODEL from .env",
    )
    p.add_argument(
        "--llm-judge-concurrency",
        type=int,
        default=DEFAULT_LLM_JUDGE_CONCURRENCY,
        help="Max concurrent LLM judge requests (default: %(default)s).",
    )
    return p.parse_args()


def _resolve_llm_judge_base_url(cli_value: str | None, *, required: bool) -> str:
    explicit = (cli_value or "").strip().rstrip("/")
    if explicit:
        return explicit
    return get_llm_judge_base_url(required=required)


def _resolve_llm_judge_model(cli_value: str | None) -> str:
    explicit = (cli_value or "").strip()
    if explicit:
        return explicit
    return get_llm_judge_model()


def main() -> None:
    args = parse_args()
    try:
        rag_base_url = _resolve_rag_base_url(args.rag_base_url or None)
        collection_base = _resolve_collection_base(args.collection_base or None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    paths = _iter_jsonl_paths(list(args.gold))
    if not paths:
        raise SystemExit("No gold JSONL files resolved from --gold.")

    rows = _load_rows(paths)
    if not rows:
        raise SystemExit("No rows loaded from gold files.")

    limit = None if int(args.limit) <= 0 else int(args.limit)
    recall_ks = parse_recall_ks(str(args.recall_at_k)) or [5, 10, 40]
    request_retrieval_hits = not bool(args.skip_retrieval_hits)
    enable_llm_judge = bool(args.enable_llm_judge)

    llm_judge_base_url = ""
    llm_judge_model = ""
    if enable_llm_judge:
        try:
            llm_judge_base_url = _resolve_llm_judge_base_url(args.llm_judge_base_url or None, required=True)
            llm_judge_model = _resolve_llm_judge_model(args.llm_judge_model or None)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    results = asyncio.run(
        _evaluate_all(
            rows,
            rag_base_url=rag_base_url,
            collection_base=collection_base,
            k=int(args.k),
            k_max=int(args.k_max),
            concurrency=int(args.concurrency),
            limit=limit,
            request_retrieval_hits=request_retrieval_hits,
            recall_ks=recall_ks,
            enable_llm_judge=enable_llm_judge,
            llm_judge_base_url=llm_judge_base_url,
            llm_judge_model=llm_judge_model,
            llm_judge_api_key=get_llm_judge_api_key(),
            llm_judge_concurrency=int(args.llm_judge_concurrency),
            llm_judge_max_tokens=get_llm_judge_max_tokens(),
            llm_judge_timeout=get_llm_judge_timeout(),
        )
    )
    run_meta = build_run_metadata(
        rag_base_url=rag_base_url,
        collection_base=collection_base,
        k=int(args.k),
        k_max=int(args.k_max),
        gold_paths=paths,
        recall_ks=recall_ks,
        concurrency=int(args.concurrency),
        skip_retrieval_hits=bool(args.skip_retrieval_hits),
        enable_llm_judge=enable_llm_judge,
        llm_judge_concurrency=int(args.llm_judge_concurrency) if enable_llm_judge else None,
        llm_judge_model=llm_judge_model if enable_llm_judge else None,
        llm_judge_base_url=llm_judge_base_url if enable_llm_judge else None,
    )
    summary = summarize(results, recall_ks=recall_ks, run_meta=run_meta)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote per-row report: %s", out_path)

    if args.summary_json:
        sum_path = Path(args.summary_json)
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote summary: %s", sum_path)

    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if not baseline_path.is_file():
            raise SystemExit(f"Baseline file not found: {baseline_path}")
        baseline = load_baseline(baseline_path)
        regressions = compare_summaries(
            summary,
            baseline,
            tolerance=float(args.baseline_tolerance),
        )
        if regressions:
            for msg in regressions:
                print(msg, file=sys.stderr)
            raise SystemExit(f"Baseline regression: {len(regressions)} metric(s) below threshold.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    main()
