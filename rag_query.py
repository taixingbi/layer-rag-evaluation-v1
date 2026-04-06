"""Fill rows by calling ``POST /v1/rag/query``; async client with bounded concurrency (see tmp.md)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET = _REPO_ROOT / "dataset" / "dataset-gold-test-1.0.0.json"
_DEFAULT_OUT = _REPO_ROOT / "result" / "dataset-gold-test-1.0.0.json"
_DEFAULT_BASE = "http://127.0.0.1:8000"
_MAX_CONCURRENCY_DEFAULT = 20
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_SEC = 1.0


def _retryable_http_status(code: int) -> bool:
    return code == 408 or code == 429 or code >= 500


async def _rag_query_async(
    client: httpx.AsyncClient,
    base: str,
    *,
    question: str,
    collection_base: str,
    request_id: str,
    session_id: str,
    k: int,
    k_max: int,
    max_attempts: int,
    retry_backoff_sec: float,
) -> dict[str, Any]:
    """One async HTTP POST per call with retries; returns API body ``{question, answer, citations, meta, ...}``."""
    url = f"{base.rstrip('/')}/v1/rag/query"
    body = {
        "question": question,
        "collection_base": collection_base,
        "request_id": request_id,
        "session_id": session_id,
        "k": k,
        "k_max": k_max,
    }
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            r = await client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code >= 400:
                if _retryable_http_status(r.status_code) and attempt + 1 < attempts:
                    delay = retry_backoff_sec * (2**attempt)
                    print(
                        f"[rag_query] HTTP {r.status_code}, sleep {delay:.1f}s "
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
                        f"[rag_query] bad JSON body, sleep {delay:.1f}s "
                        f"(attempt {attempt + 1}/{attempts})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise ValueError(f"Unexpected response: {data!r}")
            return data
        except (httpx.RequestError, ValueError) as e:
            if attempt + 1 >= attempts:
                raise
            delay = retry_backoff_sec * (2**attempt)
            print(
                f"[rag_query] {type(e).__name__}: {e!s}, sleep {delay:.1f}s "
                f"(attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            await asyncio.sleep(delay)


def _apply_rag_to_row(row: dict[str, Any], data: dict[str, Any]) -> None:
    """Merge RAG response into row; keeps ``input`` / ``output``; sets ``inference-output`` from API answer."""
    ans = data.get("answer", "")
    row["citations"] = data.get("citations", []) if isinstance(data.get("citations"), list) else []
    row["meta"] = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
    row["inference-output"] = str(ans)


def _apply_error_to_row(row: dict[str, Any], message: str) -> None:
    row["citations"] = []
    row["meta"] = {}
    row["inference-output"] = message


async def _process_row(
    idx: int,
    row: dict[str, Any],
    *,
    n: int,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    collection_base: str,
    k: int,
    k_max: int,
    max_attempts: int,
    retry_backoff_sec: float,
) -> None:
    q = (row.get("input") or row.get("question") or "").strip()
    if not q:
        print(f"[{idx + 1}/{n}] (empty question, skip)", flush=True)
        row["citations"] = []
        row["meta"] = {}
        row["inference-output"] = ""
        return
    rid = f"eva-{idx}-{uuid.uuid4().hex[:8]}"
    sid = f"eva-session-{uuid.uuid4().hex[:8]}"
    async with sem:
        # Log only after taking a concurrency slot so output reflects real parallelism.
        print(f"[{idx + 1}/{n}] {q[:80]!r}...", flush=True)
        try:
            data = await _rag_query_async(
                client,
                base_url,
                question=q,
                collection_base=collection_base,
                request_id=rid,
                session_id=sid,
                k=k,
                k_max=k_max,
                max_attempts=max_attempts,
                retry_backoff_sec=retry_backoff_sec,
            )
            _apply_rag_to_row(row, data)
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:500]
            _apply_error_to_row(row, f"ERROR HTTP: {e.response.status_code} {body}")
        except (httpx.RequestError, ValueError) as e:
            _apply_error_to_row(row, f"ERROR: {e}")


async def _run_rag_async(
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    collection: str,
    k: int,
    k_max: int,
    timeout: float,
    max_concurrency: int,
    max_attempts: int,
    retry_backoff_sec: float,
) -> None:
    n = len(rows)
    cap = max(1, min(max_concurrency, n))
    sem = asyncio.Semaphore(cap)
    limits = httpx.Limits(max_connections=cap + 5, max_keepalive_connections=cap + 5)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(
            *(
                _process_row(
                    idx,
                    row,
                    n=n,
                    client=client,
                    sem=sem,
                    base_url=base_url,
                    collection_base=collection,
                    k=k,
                    k_max=k_max,
                    max_attempts=max_attempts,
                    retry_backoff_sec=retry_backoff_sec,
                )
                for idx, row in enumerate(rows)
            )
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="For each row, POST /v1/rag/query (async, bounded concurrency); citations, meta, inference-output.",
    )
    p.add_argument(
        "-i",
        "--in",
        dest="in_path",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Input JSON array of {input, inference-output, output} (gold file)",
    )
    p.add_argument(
        "-o",
        "--out",
        dest="out_path",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output path: rows plus RAG fields from the API",
    )
    p.add_argument(
        "--base-url",
        default=_DEFAULT_BASE,
        help="RAG HTTP server base (default: http://127.0.0.1:8000)",
    )
    p.add_argument("-c", "--collection", default="taixing_knowledge", help="collection_base in JSON body")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--k-max", type=int, default=40)
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds")
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=_MAX_CONCURRENCY_DEFAULT,
        metavar="N",
        help=f"Max concurrent /v1/rag/query requests (default: {_MAX_CONCURRENCY_DEFAULT})",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=_DEFAULT_MAX_ATTEMPTS,
        metavar="N",
        help=f"Max attempts per row on transient HTTP/network errors (default: {_DEFAULT_MAX_ATTEMPTS})",
    )
    p.add_argument(
        "--retry-backoff",
        type=float,
        default=_DEFAULT_RETRY_BACKOFF_SEC,
        metavar="SEC",
        help=f"Base delay in seconds before retries; doubled each retry (default: {_DEFAULT_RETRY_BACKOFF_SEC})",
    )
    args = p.parse_args(argv)

    in_path = args.in_path
    out_path = args.out_path
    if not in_path.is_file():
        print(f"Missing input file: {in_path}", file=sys.stderr)
        return 1

    with open(in_path, encoding="utf-8") as f:
        rows: list[dict] = json.load(f)

    if not isinstance(rows, list):
        print("Dataset must be a JSON array", file=sys.stderr)
        return 1

    n = len(rows)
    asyncio.run(
        _run_rag_async(
            rows,
            base_url=args.base_url,
            collection=args.collection,
            k=args.k,
            k_max=args.k_max,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
            max_attempts=max(1, args.max_attempts),
            retry_backoff_sec=max(0.0, args.retry_backoff),
        )
    )

    for row in rows:
        row.pop("question", None)
        row.pop("answer", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(out_path)
    print(f"Wrote {n} rows to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
