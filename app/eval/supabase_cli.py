"""CLI for Supabase RAG eval history (list runs / baselines)."""

from __future__ import annotations

import argparse
import json

from app.eval.supabase_store import fetch_active_baseline, list_recent_runs, supabase_configured


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query rag_eval_runs / rag_eval_baselines in Supabase.")
    p.add_argument("--env", default="dev", help="Environment label (default: dev)")
    sub = p.add_subparsers(dest="command", required=True)

    runs = sub.add_parser("list-runs", help="Recent eval runs")
    runs.add_argument("--limit", type=int, default=20)

    sub.add_parser("show-baseline", help="Active baseline for --env")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not supabase_configured(env=args.env):
        raise SystemExit(
            f"Supabase credentials required for env={args.env}. "
            "Set SUPABASE_URL_<ENV> and SUPABASE_SECRET_KEY_<ENV> in .env."
        )

    if args.command == "list-runs":
        rows = list_recent_runs(env=args.env, limit=int(args.limit))
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    if args.command == "show-baseline":
        row = fetch_active_baseline(env=args.env)
        if not row:
            raise SystemExit(f"No active baseline for env={args.env}")
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
