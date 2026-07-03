"""Unit tests for eval scoring (no live RAG)."""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.baseline import compare_summaries
from app.eval.scoring import (
    gold_chunk_id,
    heuristic_quality,
    hits_by_stage,
    must_contain_hits,
    parse_recall_ks,
    rank_of,
    retrieval_row_fields,
    summarize,
)


def test_parse_recall_ks_dedupes_and_sorts():
    assert parse_recall_ks("10, 5, 10, 40") == [5, 10, 40]
    assert parse_recall_ks("bad,0,-1") == []


def test_must_contain_hits_case_insensitive():
    hits, total, missing = must_contain_hits("Taixing works at Saks.", ["saks", "missing"])
    assert total == 2
    assert hits == 1
    assert missing == ["missing"]


FIXTURES = Path(__file__).parent / "fixtures"
BASELINE_FIXTURE = FIXTURES / "baseline_summary.json"
GOLD_UUID = "08d53b51-7e9f-58db-8863-fc437b7100fe"


def test_gold_chunk_id_requires_uuid():
    assert gold_chunk_id({"id": GOLD_UUID}) == GOLD_UUID
    assert gold_chunk_id({"id": "multi-hop-1"}) is None


def test_retrieval_row_fields_finds_rank():
    row = {"id": GOLD_UUID}
    data = {
        "retrieval_hits": [
            {"stage": "retrieve", "rank": 2, "chunk_id": "other"},
            {"stage": "retrieve", "rank": 1, "chunk_id": GOLD_UUID},
            {"stage": "rerank", "rank": 1, "chunk_id": GOLD_UUID},
        ]
    }
    out = retrieval_row_fields(row, data, request_retrieval_hits=True, recall_ks=[5])
    assert out["retrieval_scored"] is True
    assert out["rank_retrieve"] == 1
    assert out["rank_rerank"] == 1
    assert out["hit_retrieve_at_5"] is True


def test_heuristic_quality_uses_must_contain_and_citations():
    hq = heuristic_quality(
        cite_sources={"personal_profile"},
        must_contain_pass=True,
        gold_source_hit_val=True,
        required_sources_pass=None,
    )
    assert hq["heuristic_quality"]["correct"] is True
    assert hq["heuristic_quality"]["faithful"] is True
    assert hq["heuristic_quality_score"] == 1.0


def test_summarize_heuristic_quality_keys():
    results = [
        {
            "ok": True,
            "must_contain_pass": True,
            "must_contain_total": 1,
            "heuristic_quality": {
                "correct": True,
                "faithful": False,
                "complete": True,
                "precise": False,
                "cited": False,
            },
            "heuristic_quality_score": 0.4,
            "retrieval_scored": True,
            "rr_retrieve": 1.0,
            "rr_rerank": 0.5,
            "rank_retrieve": 1,
            "rank_rerank": 2,
            "hit_retrieve_at_5": True,
            "hit_rerank_at_5": True,
            "precision_at_5_retrieve": 0.2,
            "precision_at_5_rerank": 0.2,
            "ndcg_at_5_retrieve": 1.0,
            "ndcg_at_5_rerank": 1.0,
            "f1_at_5_retrieve": 0.33,
            "f1_at_5_rerank": 0.33,
            "latency_ms_total": 1000,
        }
    ]
    summary = summarize(results, recall_ks=[5])
    assert summary["rows"] == 1
    assert summary["heuristic_quality_scored_rows"] == 1
    assert "heuristic_quality_correct_pass" in summary
    assert summary["mrr_retrieve"] == 1.0
    assert summary["must_contain_pass_rate"] == 1.0


def test_baseline_fixture_loads():
    baseline = json.loads(BASELINE_FIXTURE.read_text(encoding="utf-8"))
    assert baseline["rag_calls_failed"] == 0
    assert "heuristic_quality_score_mean" in baseline
    assert "description" in baseline


def test_baseline_regression_detects_drop():
    baseline = {"recall_at_5_rerank": 0.9, "rag_calls_failed": 0}
    current_ok = {"recall_at_5_rerank": 0.88, "rag_calls_failed": 0}
    assert compare_summaries(current_ok, baseline, tolerance=0.05) == []

    current_bad = {"recall_at_5_rerank": 0.80, "rag_calls_failed": 0}
    msgs = compare_summaries(current_bad, baseline, tolerance=0.05)
    assert any("recall_at_5_rerank" in m for m in msgs)


def test_hits_by_stage_orders_by_rank():
    stages = hits_by_stage(
        [
            {"stage": "retrieve", "rank": 3, "chunk_id": "c"},
            {"stage": "retrieve", "rank": 1, "chunk_id": "a"},
        ]
    )
    assert stages["retrieve"] == ["a", "c"]
    assert rank_of(stages["retrieve"], "c") == 2
