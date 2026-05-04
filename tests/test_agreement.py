"""Tests for coding_eval.agreement."""

import pytest

from coding_eval.agreement import _binary_confusion, _per_code_metrics, compute_agreement
from coding_eval.models import CodedSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seg(sid, human, llm=None):
    return CodedSegment(segment_id=sid, text=f"text_{sid}", human_codes=human, llm_codes=llm)


def perfect_segs():
    """Five segments with exact human == llm agreement."""
    return [
        seg("s1", ["a", "b"], ["a", "b"]),
        seg("s2", ["b"],      ["b"]),
        seg("s3", ["a"],      ["a"]),
        seg("s4", ["a", "b"], ["a", "b"]),
        seg("s5", ["b"],      ["b"]),
    ]


def no_agree_segs():
    """Five segments where LLM always picks the wrong code."""
    return [
        seg("s1", ["a"], ["b"]),
        seg("s2", ["a"], ["b"]),
        seg("s3", ["b"], ["a"]),
        seg("s4", ["b"], ["a"]),
        seg("s5", ["a"], ["b"]),
    ]


# ---------------------------------------------------------------------------
# _binary_confusion
# ---------------------------------------------------------------------------


class TestBinaryConfusion:
    def test_all_true_positives(self):
        segs = [seg("s1", ["a"], ["a"]), seg("s2", ["a"], ["a"])]
        tp, fp, fn, tn = _binary_confusion(segs, "a")
        assert tp == 2
        assert fp == 0
        assert fn == 0
        assert tn == 0

    def test_all_true_negatives(self):
        segs = [seg("s1", ["b"], ["b"]), seg("s2", ["b"], ["b"])]
        tp, fp, fn, tn = _binary_confusion(segs, "a")
        assert tp == 0
        assert fp == 0
        assert fn == 0
        assert tn == 2

    def test_false_positive(self):
        segs = [seg("s1", ["b"], ["a"])]
        tp, fp, fn, tn = _binary_confusion(segs, "a")
        assert fp == 1

    def test_false_negative(self):
        segs = [seg("s1", ["a"], ["b"])]
        tp, fp, fn, tn = _binary_confusion(segs, "a")
        assert fn == 1

    def test_none_llm_treated_as_negative(self):
        segs = [seg("s1", ["a"], None)]
        tp, fp, fn, tn = _binary_confusion(segs, "a")
        assert fn == 1
        assert tp == 0

    def test_sum_equals_n(self):
        segs = perfect_segs()
        for code in ["a", "b"]:
            tp, fp, fn, tn = _binary_confusion(segs, code)
            assert tp + fp + fn + tn == len(segs)


# ---------------------------------------------------------------------------
# compute_agreement — empty
# ---------------------------------------------------------------------------


class TestComputeAgreementEdgeCases:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="no segments"):
            compute_agreement([])

    def test_single_segment_perfect(self):
        segs = [seg("s1", ["a"], ["a"])]
        metrics = compute_agreement(segs)
        assert metrics.total_segments == 1
        assert metrics.percent_agreement == 1.0

    def test_single_segment_no_agree(self):
        segs = [seg("s1", ["a"], ["b"])]
        metrics = compute_agreement(segs)
        assert metrics.percent_agreement == 0.0

    def test_none_llm_codes_treated_as_empty(self):
        segs = [seg("s1", ["a"], None), seg("s2", ["b"], None)]
        metrics = compute_agreement(segs)
        assert metrics.percent_agreement == 0.0

    def test_empty_human_and_empty_llm_codes(self):
        segs = [seg("s1", [], [])]
        # No codes at all — no per-code metrics, percent_agreement=1.0 (exact match ∅==∅)
        metrics = compute_agreement(segs)
        assert metrics.percent_agreement == 1.0
        assert metrics.per_code == []


# ---------------------------------------------------------------------------
# compute_agreement — perfect agreement
# ---------------------------------------------------------------------------


class TestPerfectAgreement:
    def setup_method(self):
        self.metrics = compute_agreement(perfect_segs())

    def test_percent_agreement_is_one(self):
        assert self.metrics.percent_agreement == 1.0

    def test_kappa_is_one(self):
        assert abs(self.metrics.cohens_kappa - 1.0) < 1e-6

    def test_macro_f1_is_one(self):
        assert abs(self.metrics.macro_f1 - 1.0) < 1e-6

    def test_weighted_f1_is_one(self):
        assert abs(self.metrics.weighted_f1 - 1.0) < 1e-6

    def test_grade_is_a(self):
        assert self.metrics.grade == "A"

    def test_per_code_precision_one(self):
        for pc in self.metrics.per_code:
            assert abs(pc.precision - 1.0) < 1e-6

    def test_per_code_recall_one(self):
        for pc in self.metrics.per_code:
            assert abs(pc.recall - 1.0) < 1e-6

    def test_per_code_f1_one(self):
        for pc in self.metrics.per_code:
            assert abs(pc.f1 - 1.0) < 1e-6

    def test_per_code_kappa_one(self):
        for pc in self.metrics.per_code:
            assert abs(pc.cohens_kappa - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_agreement — no agreement
# ---------------------------------------------------------------------------


class TestNoAgreement:
    def setup_method(self):
        self.metrics = compute_agreement(no_agree_segs())

    def test_percent_agreement_is_zero(self):
        assert self.metrics.percent_agreement == 0.0

    def test_kappa_is_low(self):
        assert self.metrics.cohens_kappa < 0.5

    def test_grade_is_not_a(self):
        assert self.metrics.grade != "A"

    def test_per_code_f1_low(self):
        for pc in self.metrics.per_code:
            assert pc.f1 < 0.5


# ---------------------------------------------------------------------------
# compute_agreement — structural / numeric correctness
# ---------------------------------------------------------------------------


class TestAgreementStructure:
    def test_total_segments_correct(self):
        segs = perfect_segs()
        metrics = compute_agreement(segs)
        assert metrics.total_segments == len(segs)

    def test_per_code_count(self):
        segs = [
            seg("s1", ["a", "b"], ["a", "b"]),
            seg("s2", ["c"],      ["c"]),
        ]
        metrics = compute_agreement(segs)
        assert len(metrics.per_code) == 3  # a, b, c

    def test_per_code_sorted_alphabetically(self):
        segs = [
            seg("s1", ["z", "m", "a"], ["z", "m", "a"]),
        ]
        metrics = compute_agreement(segs)
        codes = [pc.code for pc in metrics.per_code]
        assert codes == sorted(codes)

    def test_codes_only_in_llm_included(self):
        segs = [seg("s1", ["a"], ["a", "x"])]
        metrics = compute_agreement(segs)
        codes = [pc.code for pc in metrics.per_code]
        assert "x" in codes

    def test_codes_only_in_human_included(self):
        segs = [seg("s1", ["a", "y"], ["a"])]
        metrics = compute_agreement(segs)
        codes = [pc.code for pc in metrics.per_code]
        assert "y" in codes

    def test_kappa_in_valid_range(self):
        segs = perfect_segs() + no_agree_segs()
        metrics = compute_agreement(segs)
        assert -1.0 <= metrics.cohens_kappa <= 1.0

    def test_percent_agreement_in_range(self):
        segs = perfect_segs()
        metrics = compute_agreement(segs)
        assert 0.0 <= metrics.percent_agreement <= 1.0

    def test_macro_f1_in_range(self):
        segs = perfect_segs()
        metrics = compute_agreement(segs)
        assert 0.0 <= metrics.macro_f1 <= 1.0

    def test_weighted_f1_in_range(self):
        segs = perfect_segs()
        metrics = compute_agreement(segs)
        assert 0.0 <= metrics.weighted_f1 <= 1.0

    def test_per_code_support_sums_correctly(self):
        segs = [
            seg("s1", ["a"], ["a"]),
            seg("s2", ["a", "b"], ["a"]),
            seg("s3", ["b"], ["b"]),
        ]
        metrics = compute_agreement(segs)
        support_map = {pc.code: pc.support for pc in metrics.per_code}
        assert support_map["a"] == 2
        assert support_map["b"] == 2

    def test_multi_label_percent_agreement(self):
        segs = [
            seg("s1", ["a", "b"], ["a", "b"]),   # exact match
            seg("s2", ["a", "b"], ["a"]),         # mismatch
        ]
        metrics = compute_agreement(segs)
        assert metrics.percent_agreement == 0.5

    def test_grade_interpretation_present(self):
        segs = perfect_segs()
        metrics = compute_agreement(segs)
        assert len(metrics.interpretation) > 0

    def test_to_dict_serialisable(self):
        import json
        segs = perfect_segs()
        metrics = compute_agreement(segs)
        d = metrics.to_dict()
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# compute_agreement — precision / recall correctness
# ---------------------------------------------------------------------------


class TestPrecisionRecall:
    def test_precision_one_no_false_positives(self):
        # LLM only predicts a when human also says a
        segs = [
            seg("s1", ["a"], ["a"]),
            seg("s2", ["b"], ["b"]),
        ]
        metrics = compute_agreement(segs)
        a_metrics = next(pc for pc in metrics.per_code if pc.code == "a")
        assert abs(a_metrics.precision - 1.0) < 1e-6

    def test_recall_one_no_false_negatives(self):
        segs = [
            seg("s1", ["a"], ["a"]),
            seg("s2", ["a"], ["a"]),
        ]
        metrics = compute_agreement(segs)
        a_metrics = next(pc for pc in metrics.per_code if pc.code == "a")
        assert abs(a_metrics.recall - 1.0) < 1e-6

    def test_precision_zero_all_false_positives(self):
        segs = [
            seg("s1", ["b"], ["a"]),  # a is FP
        ]
        metrics = compute_agreement(segs)
        a_metrics = next(pc for pc in metrics.per_code if pc.code == "a")
        assert a_metrics.precision == 0.0

    def test_recall_zero_all_false_negatives(self):
        segs = [
            seg("s1", ["a"], ["b"]),  # a is FN
        ]
        metrics = compute_agreement(segs)
        a_metrics = next(pc for pc in metrics.per_code if pc.code == "a")
        assert a_metrics.recall == 0.0

    def test_f1_zero_when_no_tp(self):
        segs = [
            seg("s1", ["a"], ["b"]),
        ]
        metrics = compute_agreement(segs)
        a_metrics = next(pc for pc in metrics.per_code if pc.code == "a")
        assert a_metrics.f1 == 0.0
