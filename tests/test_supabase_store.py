"""Tests for Supabase eval store (mocked supabase-py client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.eval.supabase_store import (
    baseline_row_to_compare_dict,
    fetch_active_baseline,
    record_run,
    summary_to_run_row,
    supabase_configured,
)


def test_summary_to_run_row_maps_metrics():
    summary = {
        "rows": 20,
        "rag_calls_failed": 0,
        "mrr_rerank": 0.8,
        "recall_at_5_rerank": 0.85,
        "llm_judge_score_mean": 0.75,
        "latency_ms_p95": 3000,
        "run_meta": {
            "git_sha": "abc",
            "gold_dataset_sha256": "goldsha",
            "collection_base": "taixing_knowledge",
            "gold_rows_loaded": 196,
        },
    }
    row = summary_to_run_row(
        summary,
        env="dev",
        baseline_id=None,
        passed=True,
        report_storage_path="/tmp/report.json",
        notes=None,
    )
    assert row["env"] == "dev"
    assert row["pass"] is True
    assert row["mrr_rerank"] == 0.8
    assert row["gold_dataset_sha256"] == "goldsha"


def test_baseline_row_to_compare_dict():
    row = {
        "mrr_rerank": 0.75,
        "recall_at_5_rerank": 0.87,
        "gold_dataset_sha256": "abc",
        "baseline_json": {"llm_judge_score_mean": 0.7},
    }
    out = baseline_row_to_compare_dict(row)
    assert out["mrr_rerank"] == 0.75
    assert out["llm_judge_score_mean"] == 0.7
    assert out["run_meta"]["gold_dataset_sha256"] == "abc"


def _mock_supabase_client(*, insert_data: list[dict] | None = None, select_data: list[dict] | None = None):
    client = MagicMock()
    table = MagicMock()
    client.table.return_value = table

    if insert_data is not None:
        table.insert.return_value.execute.return_value = MagicMock(data=insert_data)

    if select_data is not None:
        execute_result = MagicMock(data=select_data)
        chain = MagicMock()
        chain.execute.return_value = execute_result
        chain.eq.return_value = chain
        chain.order.return_value = chain
        chain.limit.return_value = chain
        table.select.return_value = chain

    return client


def test_record_run_inserts_via_supabase_client():
    summary = {"rows": 1, "rag_calls_failed": 0, "run_meta": {"git_sha": "x"}}
    client = _mock_supabase_client(insert_data=[{"id": "run-uuid-1"}])

    out = record_run(summary, env="dev", baseline_id=None, passed=True, client=client)
    assert out["id"] == "run-uuid-1"
    client.table.assert_called_with("rag_eval_runs")
    insert_row = client.table.return_value.insert.call_args.args[0]
    assert insert_row["env"] == "dev"
    assert insert_row["pass"] is True


def test_fetch_active_baseline():
    client = _mock_supabase_client(select_data=[{"id": "b1", "name": "dev-main", "mrr_rerank": 0.7}])

    row = fetch_active_baseline(env="dev", client=client)
    assert row["id"] == "b1"
    client.table.assert_called_with("rag_eval_baselines")


def test_supabase_configured_false_when_missing():
    with patch("app.eval.supabase_store.get_supabase_url", side_effect=ValueError("missing")):
        assert supabase_configured() is False
