"""
Deterministic tests for the retrieval evaluation harness.

The fixtures build synthetic ``SearchResult`` dataclass objects (the type the
retrieval strategies return) and verify Hit Rate / MRR metric values against
manually computed expectations.

This module does NOT touch Elasticsearch, the retrieval strategies, or the
reranker. It exercises the metric functions directly plus the error-isolation
behavior of ``evaluate_retrieval``.
"""

import json
import os

import pytest

from domain.models import SearchResult
from evaluation.retrieval_eval import (
    hit_rate_at_k,
    mrr_at_k,
    evaluate_retrieval,
)
from evaluation.fixtures import (
    make_search_result,
    results_with_expected_at,
    no_hit_results,
    dict_results_with_expected_at,
)


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------

class TestHitRateAtK:
    """Expected chunk_id found ⇒ 1.0; otherwise 0.0 (binary)."""

    def test_returns_float1_when_expected_is_present(self):
        results = results_with_expected_at(rank=1)
        assert hit_rate_at_k(results, "expected", k=5) == 1.0

    def test_returns_float0_when_expected_is_absent(self):
        results = no_hit_results()
        assert hit_rate_at_k(results, "missing", k=5) == 0.0

    def test_rank1_hits_every_k(self):
        results = results_with_expected_at(rank=1)
        for k in (1, 3, 5, 10):
            assert hit_rate_at_k(results, "expected", k=k) == 1.0

    def test_rank3_hits_k3_k5_k10_not_k1(self):
        results = results_with_expected_at(rank=3)
        assert hit_rate_at_k(results, "expected", k=1) == 0.0
        assert hit_rate_at_k(results, "expected", k=3) == 1.0
        assert hit_rate_at_k(results, "expected", k=5) == 1.0
        assert hit_rate_at_k(results, "expected", k=10) == 1.0

    def test_rank5_hits_k5_k10_not_k1_k3(self):
        results = results_with_expected_at(rank=5)
        assert hit_rate_at_k(results, "expected", k=1) == 0.0
        assert hit_rate_at_k(results, "expected", k=3) == 0.0
        assert hit_rate_at_k(results, "expected", k=5) == 1.0
        assert hit_rate_at_k(results, "expected", k=10) == 1.0

    def test_rank8_hits_k10_not_k5(self):
        results = results_with_expected_at(rank=8)
        assert hit_rate_at_k(results, "expected", k=1) == 0.0
        assert hit_rate_at_k(results, "expected", k=3) == 0.0
        assert hit_rate_at_k(results, "expected", k=5) == 0.0
        assert hit_rate_at_k(results, "expected", k=10) == 1.0

    def test_expected_rank_beyond_total_is_a_miss(self):
        results = results_with_expected_at(rank=12, total=10)  # rank > list length
        assert hit_rate_at_k(results, "expected", k=10) == 0.0

    def test_multiple_relevant_any_match_is_hit(self):
        results = results_with_expected_at(rank=3, total=10)
        # Two relevant ids: "expected" at rank 3, plus an additional one.
        results.append(make_search_result("expected_b", rank=11, score=0.0))
        assert hit_rate_at_k(results, ["expected", "expected_b"], k=3) == 1.0
        assert hit_rate_at_k(results, ["expected", "expected_b"], k=1) == 0.0
        # Only the rank-3 relevant id determines the result in top-3.
        results2 = results_with_expected_at(rank=3, total=10)
        assert hit_rate_at_k(results2, ["expected_a", "found_later"], k=5) == 0.0

    def test_empty_results_never_hits(self):
        assert hit_rate_at_k([], "expected", k=5) == 0.0

    def test_empty_expected_ids_is_a_miss(self):
        results = results_with_expected_at(rank=1)
        assert hit_rate_at_k(results, [], k=5) == 0.0


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------

class TestMrrAtK:
    """Reciprocal rank of first (highest-ranked) relevant result in top-K."""

    def test_rank1_gives_mrr_1(self):
        results = results_with_expected_at(rank=1)
        assert mrr_at_k(results, "expected", k=5) == pytest.approx(1.0)

    def test_rank3_gives_mrr_one_third(self):
        results = results_with_expected_at(rank=3)
        assert mrr_at_k(results, "expected", k=5) == pytest.approx(1.0 / 3.0)
        assert mrr_at_k(results, "expected", k=10) == pytest.approx(1.0 / 3.0)

    def test_rank3_is_miss_for_mrr1(self):
        results = results_with_expected_at(rank=3)
        assert mrr_at_k(results, "expected", k=1) == 0.0

    def test_rank5_gives_mrr_point2(self):
        results = results_with_expected_at(rank=5)
        assert mrr_at_k(results, "expected", k=5) == pytest.approx(0.2)
        assert mrr_at_k(results, "expected", k=10) == pytest.approx(0.2)

    def test_rank8_misses_mrr5_hits_mrr10(self):
        results = results_with_expected_at(rank=8)
        assert mrr_at_k(results, "expected", k=5) == 0.0
        assert mrr_at_k(results, "expected", k=10) == pytest.approx(1.0 / 8.0)

    def test_no_relevant_is_zero(self):
        results = no_hit_results()
        assert mrr_at_k(results, "missing", k=10) == 0.0

    def test_relevant_beyond_topk_is_zero(self):
        results = results_with_expected_at(rank=12, total=10)
        assert mrr_at_k(results, "expected", k=10) == 0.0

    def test_multiple_relevant_uses_highest_rank(self):
        # Highest-ranked relevant (rank 3) determines MRR@5 and MRR@10.
        results = results_with_expected_at(rank=3, total=10)
        results[5] = make_search_result("expected_b", rank=6, score=4.0)
        assert mrr_at_k(results, ["expected", "expected_b"], k=5) == pytest.approx(1.0 / 3.0)
        assert mrr_at_k(results, ["expected", "expected_b"], k=10) == pytest.approx(1.0 / 3.0)

    def test_empty_results_is_zero(self):
        assert mrr_at_k([], "expected", k=5) == 0.0


# ---------------------------------------------------------------------------
# SearchResult / dict compatibility
# ---------------------------------------------------------------------------

class TestResultFormatCompatibility:
    def test_search_result_dataclass_supported(self):
        results = results_with_expected_at(rank=3)
        assert all(isinstance(r, SearchResult) for r in results)
        assert hit_rate_at_k(results, "expected", k=3) == 1.0
        assert mrr_at_k(results, "expected", k=5) == pytest.approx(1.0 / 3.0)

    def test_dict_style_results_still_supported(self):
        results = dict_results_with_expected_at(rank=3)
        assert hit_rate_at_k(results, "expected", k=3) == 1.0
        assert hit_rate_at_k(results, "expected", k=1) == 0.0
        assert mrr_at_k(results, "expected", k=5) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# evaluate_retrieval error isolation (no real search)
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalErrorIsolation:
    def test_retrieval_failure_is_not_silent_zero(self, tmp_path, monkeypatch):
        """Search failures must surface as errors, never as 0.0 metrics."""
        gt_file = tmp_path / "ground_truth.json"
        gt_file.write_text(
            json.dumps([
                {"question": "Q1", "chunk_id": "c1", "expected_answer": "a1"},
                {"question": "Q2", "chunk_id": "c2", "expected_answer": "a2"},
            ]),
            encoding="utf-8",
        )

        def boom(*args, **kwargs):
            raise RuntimeError("simulated search failure")

        monkeypatch.setattr("evaluation.retrieval_eval.SearchStrategyFactory.get_strategy", boom)
        monkeypatch.setattr("evaluation.retrieval_eval.CrossEncoderReranker", boom)

        out_file = tmp_path / "results.json"
        summary = evaluate_retrieval(
            ground_truth_path=str(gt_file), output_path=str(out_file)
        )

        for method in ("vector", "text", "hybrid", "hybrid_rerank"):
            m = summary[method]
            assert m["n_successful"] == 0, f"{method} should have 0 successful queries"
            assert m["n_errors"] == 2, f"{method} should record 2 errors, got {m['n_errors']}"
            assert m["status"] == "no_successful_queries"
            # Metrics must NOT exist — failures can never look like zero scores.
            assert "hit_rate" not in m
            assert "mrr" not in m
            assert "hit_rate_at_5" not in m
            assert len(m["errors"]) == 2
            assert "simulated search failure" in m["errors"][0]["error"]
            assert m["errors"][0]["query_index"] == 0

        assert summary["best_method"] is None
        assert out_file.exists()

    def test_ground_truth_missing_returns_empty(self, tmp_path):
        assert evaluate_retrieval(
            ground_truth_path=str(tmp_path / "does_not_exist.json")
        ) == {}