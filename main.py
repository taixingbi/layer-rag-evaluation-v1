"""One-shot: async ``rag_query`` then ``metric`` on the same ``-o`` file (RAG + metrics)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_GOLD = _REPO_ROOT / "dataset" / "dataset-gold-test-1.0.0.json"
_DEFAULT_RAG_OUT = _REPO_ROOT / "result" / "dataset-gold-test-1.0.0.json"


def run(
    gold_path: Path,
    rag_out: Path,
    *,
    base_url: str,
    collection: str,
    k: int,
    k_max: int,
    timeout: float,
    rag_max_concurrency: int,
    rag_max_attempts: int,
    rag_retry_backoff: float,
    metric_heuristic_only: bool,
    judge_max_attempts: int | None,
    judge_retry_backoff: float | None,
    judge_max_concurrency: int,
) -> int:
    import metric as metric_mod
    import rag_query as rag_mod

    if not gold_path.is_file():
        print(f"Missing gold dataset: {gold_path}", file=sys.stderr)
        return 1
    print(
        f"[rag_query] start  {gold_path} → {rag_out}  "
        f"base={base_url}  rag_concurrency={rag_max_concurrency}  timeout={timeout}s",
        flush=True,
    )
    t0 = time.perf_counter()
    rc = rag_mod.main(
        [
            "-i",
            str(gold_path),
            "-o",
            str(rag_out),
            "--base-url",
            base_url,
            "-c",
            collection,
            "-k",
            str(k),
            "--k-max",
            str(k_max),
            "--timeout",
            str(timeout),
            "--max-concurrency",
            str(rag_max_concurrency),
            "--max-attempts",
            str(rag_max_attempts),
            "--retry-backoff",
            str(rag_retry_backoff),
        ]
    )
    rag_elapsed = time.perf_counter() - t0
    print(f"[rag_query] finish elapsed {rag_elapsed:.2f}s", flush=True)
    if rc != 0:
        return rc
    t1 = time.perf_counter()
    margv = ["-i", str(rag_out), "-o", str(rag_out)]
    if metric_heuristic_only:
        margv.append("--heuristic-only")
    if judge_max_attempts is not None:
        margv.extend(["--max-attempts", str(judge_max_attempts)])
    if judge_retry_backoff is not None:
        margv.extend(["--retry-backoff", str(judge_retry_backoff)])
    margv.extend(["--max-concurrency", str(max(1, judge_max_concurrency))])
    rc = metric_mod.main(margv)
    print(f"[metric]    finish elapsed {time.perf_counter() - t1:.2f}s", flush=True)
    return rc


if __name__ == "__main__":
    from rag_query import (
        _DEFAULT_BASE,
        _DEFAULT_MAX_ATTEMPTS,
        _DEFAULT_RETRY_BACKOFF_SEC,
        _MAX_CONCURRENCY_DEFAULT,
    )

    p = argparse.ArgumentParser(description="One shot: async RAG fill then attach metrics.")
    p.add_argument(
        "-i",
        type=Path,
        default=_DEFAULT_GOLD,
        help="Gold JSON (default: dataset/dataset-gold-test-1.0.0.json)",
    )
    p.add_argument(
        "-o",
        type=Path,
        default=_DEFAULT_RAG_OUT,
        help="Output JSON: RAG fill then metrics appended in place (same file)",
    )
    p.add_argument(
        "--base-url",
        default=_DEFAULT_BASE,
        help="RAG server base for /v1/rag/query",
    )
    p.add_argument("-c", "--collection", default="taixing_knowledge")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--k-max", type=int, default=40)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--rag-max-concurrency",
        type=int,
        default=_MAX_CONCURRENCY_DEFAULT,
        metavar="N",
        dest="rag_max_concurrency",
        help=f"Max concurrent /v1/rag/query requests (default: {_MAX_CONCURRENCY_DEFAULT})",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=_DEFAULT_MAX_ATTEMPTS,
        metavar="N",
        help=f"RAG: max HTTP attempts per row on transient errors (default: {_DEFAULT_MAX_ATTEMPTS})",
    )
    p.add_argument(
        "--retry-backoff",
        type=float,
        default=_DEFAULT_RETRY_BACKOFF_SEC,
        metavar="SEC",
        help=f"RAG: base retry delay in seconds, doubled each retry (default: {_DEFAULT_RETRY_BACKOFF_SEC})",
    )
    p.add_argument(
        "--judge-max-attempts",
        type=int,
        default=None,
        metavar="N",
        help="LLM judge: max attempts (default: env LLM_JUDGE_MAX_ATTEMPTS or 3)",
    )
    p.add_argument(
        "--judge-retry-backoff",
        type=float,
        default=None,
        metavar="SEC",
        help="LLM judge: base retry delay seconds (default: env LLM_JUDGE_RETRY_BACKOFF or 1.0)",
    )
    p.add_argument(
        "--judge-max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="LLM judge: max concurrent HTTP requests (default: same as --rag-max-concurrency)",
    )
    p.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip LLM judge in metric step (string/embed heuristics only)",
    )
    args = p.parse_args()
    judge_conc = (
        args.judge_max_concurrency
        if args.judge_max_concurrency is not None
        else args.rag_max_concurrency
    )
    raise SystemExit(
        run(
            args.i,
            args.o,
            base_url=args.base_url,
            collection=args.collection,
            k=args.k,
            k_max=args.k_max,
            timeout=args.timeout,
            rag_max_concurrency=args.rag_max_concurrency,
            rag_max_attempts=max(1, args.max_attempts),
            rag_retry_backoff=max(0.0, args.retry_backoff),
            metric_heuristic_only=args.heuristic_only,
            judge_max_attempts=args.judge_max_attempts,
            judge_retry_backoff=args.judge_retry_backoff,
            judge_max_concurrency=max(1, judge_conc),
        )
    )
