"""Tests for coding_eval.models."""

import pytest

from coding_eval.models import (
    AgreementMetrics,
    CodedSegment,
    EvalResult,
    PerCodeMetrics,
    kappa_to_grade,
)


# ---------------------------------------------------------------------------
# CodedSegment
# ---------------------------------------------------------------------------


class TestCodedSegment:
    def test_basic_creation(self):
        seg = CodedSegment(segment_id="s1", text="hello", human_codes=["a", "b"])
        assert seg.segment_id == "s1"
        assert seg.text == "hello"
        assert seg.human_codes == ["a", "b"]

    def test_llm_codes_default_none(self):
        seg = CodedSegment(segment_id="s1", text="t", human_codes=[])
        assert seg.llm_codes is None

    def test_llm_codes_set(self):
        seg = CodedSegment(segment_id="s1", text="t", human_codes=["a"], llm_codes=["a", "b"])
        assert seg.llm_codes == ["a", "b"]

    def test_to_dict_structure(self):
        seg = CodedSegment("s1", "text", ["a"], ["b"])
        d = seg.to_dict()
        assert set(d.keys()) == {"segment_id", "text", "human_codes", "llm_codes"}

    def test_to_dict_values(self):
        seg = CodedSegment("s2", "foo bar", ["x", "y"], ["x"])
        d = seg.to_dict()
        assert d["segment_id"] == "s2"
        assert d["text"] == "foo bar"
        assert d["human_codes"] == ["x", "y"]
        assert d["llm_codes"] == ["x"]

    def test_to_dict_no_llm_codes(self):
        seg = CodedSegment("s3", "t", ["a"])
        d = seg.to_dict()
        assert d["llm_codes"] is None

    def test_empty_human_codes(self):
        seg = CodedSegment("s1", "t", [])
        assert seg.human_codes == []

    def test_empty_llm_codes(self):
        seg = CodedSegment("s1", "t", ["a"], [])
        assert seg.llm_codes == []


# ---------------------------------------------------------------------------
# PerCodeMetrics
# ---------------------------------------------------------------------------


class TestPerCodeMetrics:
    def test_basic_creation(self):
        pc = PerCodeMetrics(
            code="satisfaction", precision=0.9, recall=0.8, f1=0.85,
            support=10, cohens_kappa=0.75
        )
        assert pc.code == "satisfaction"
        assert pc.precision == 0.9

    def test_to_dict_keys(self):
        pc = PerCodeMetrics("a", 0.5, 0.6, 0.55, 5, 0.4)
        d = pc.to_dict()
        assert set(d.keys()) == {"code", "precision", "recall", "f1", "support", "cohens_kappa"}

    def test_to_dict_rounded(self):
        pc = PerCodeMetrics("a", 0.333333, 0.666667, 0.444444, 3, 0.123456)
        d = pc.to_dict()
        assert d["precision"] == 0.3333
        assert d["recall"] == 0.6667
        assert d["f1"] == 0.4444
        assert d["cohens_kappa"] == 0.1235

    def test_zero_values(self):
        pc = PerCodeMetrics("x", 0.0, 0.0, 0.0, 0, 0.0)
        d = pc.to_dict()
        assert d["f1"] == 0.0


# ---------------------------------------------------------------------------
# kappa_to_grade
# ---------------------------------------------------------------------------


class TestKappaToGrade:
    @pytest.mark.parametrize("kappa,expected_grade", [
        (1.00, "A"),
        (0.85, "A"),
        (0.80, "A"),
        (0.79, "B"),
        (0.65, "B"),
        (0.60, "B"),
        (0.59, "C"),
        (0.50, "C"),
        (0.40, "C"),
        (0.39, "D"),
        (0.25, "D"),
        (0.20, "D"),
        (0.19, "F"),
        (0.00, "F"),
        (-0.10, "F"),
    ])
    def test_grade_boundaries(self, kappa, expected_grade):
        grade, _ = kappa_to_grade(kappa)
        assert grade == expected_grade

    def test_returns_tuple(self):
        result = kappa_to_grade(0.7)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_interpretation_non_empty(self):
        for k in [0.9, 0.7, 0.5, 0.3, 0.1]:
            _, interp = kappa_to_grade(k)
            assert isinstance(interp, str)
            assert len(interp) > 0


# ---------------------------------------------------------------------------
# AgreementMetrics
# ---------------------------------------------------------------------------


def _make_metrics(kappa=0.75, pct=0.8, mf1=0.7, wf1=0.72, n=10):
    pc = PerCodeMetrics("a", 0.8, 0.7, 0.75, 5, kappa)
    return AgreementMetrics(
        cohens_kappa=kappa,
        percent_agreement=pct,
        macro_f1=mf1,
        weighted_f1=wf1,
        per_code=[pc],
        total_segments=n,
    )


class TestAgreementMetrics:
    def test_grade_assigned_on_init(self):
        m = _make_metrics(kappa=0.85)
        assert m.grade == "A"

    def test_grade_b(self):
        m = _make_metrics(kappa=0.65)
        assert m.grade == "B"

    def test_grade_c(self):
        m = _make_metrics(kappa=0.45)
        assert m.grade == "C"

    def test_grade_d(self):
        m = _make_metrics(kappa=0.25)
        assert m.grade == "D"

    def test_grade_f(self):
        m = _make_metrics(kappa=0.10)
        assert m.grade == "F"

    def test_interpretation_set(self):
        m = _make_metrics(kappa=0.85)
        assert "almost" in m.interpretation.lower()

    def test_to_dict_keys(self):
        m = _make_metrics()
        d = m.to_dict()
        expected = {
            "cohens_kappa", "percent_agreement", "macro_f1", "weighted_f1",
            "grade", "interpretation", "total_segments", "per_code"
        }
        assert set(d.keys()) == expected

    def test_to_dict_per_code_is_list(self):
        m = _make_metrics()
        d = m.to_dict()
        assert isinstance(d["per_code"], list)

    def test_to_dict_values_rounded(self):
        m = _make_metrics(kappa=0.123456, pct=0.567891)
        d = m.to_dict()
        assert d["cohens_kappa"] == 0.1235
        assert d["percent_agreement"] == 0.5679

    def test_total_segments(self):
        m = _make_metrics(n=42)
        assert m.total_segments == 42


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_basic_creation(self):
        m = _make_metrics()
        seg = CodedSegment("s1", "t", ["a"])
        r = EvalResult(segments=[seg], metrics=m)
        assert r.narrative is None
        assert r.llm_backend is None

    def test_to_dict_keys(self):
        m = _make_metrics()
        seg = CodedSegment("s1", "t", ["a"])
        r = EvalResult(segments=[seg], metrics=m, narrative="ok", llm_backend="openai", model="openai/gpt-4o-mini")
        d = r.to_dict()
        assert set(d.keys()) == {"llm_backend", "model", "metrics", "narrative", "segments"}

    def test_to_dict_segments_list(self):
        m = _make_metrics()
        seg = CodedSegment("s1", "t", ["a"])
        r = EvalResult(segments=[seg], metrics=m)
        d = r.to_dict()
        assert isinstance(d["segments"], list)
        assert len(d["segments"]) == 1

    def test_narrative_in_dict(self):
        m = _make_metrics()
        r = EvalResult(segments=[], metrics=m, narrative="Test narrative")
        assert r.to_dict()["narrative"] == "Test narrative"

    def test_no_narrative_null_in_dict(self):
        m = _make_metrics()
        r = EvalResult(segments=[], metrics=m)
        assert r.to_dict()["narrative"] is None
