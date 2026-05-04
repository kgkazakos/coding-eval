"""Data models for coding-eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodedSegment:
    """A single transcript or research segment with applied codes."""

    segment_id: str
    text: str
    human_codes: List[str]
    llm_codes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "human_codes": self.human_codes,
            "llm_codes": self.llm_codes,
        }


@dataclass
class PerCodeMetrics:
    """Agreement metrics for a single thematic code."""

    code: str
    precision: float
    recall: float
    f1: float
    support: int
    cohens_kappa: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "cohens_kappa": round(self.cohens_kappa, 4),
        }


# Landis & Koch (1977) Kappa interpretation thresholds
_GRADE_THRESHOLDS = [
    (0.80, "A", "Almost perfect agreement"),
    (0.60, "B", "Substantial agreement"),
    (0.40, "C", "Moderate agreement"),
    (0.20, "D", "Fair agreement"),
    (-1.01, "F", "Slight or poor agreement"),
]


def kappa_to_grade(kappa: float) -> tuple[str, str]:
    """Return (letter_grade, interpretation) for a kappa value."""
    for threshold, grade, interpretation in _GRADE_THRESHOLDS:
        if kappa >= threshold:
            return grade, interpretation
    return "F", "Slight or poor agreement"


@dataclass
class AgreementMetrics:
    """Aggregate inter-rater agreement metrics."""

    cohens_kappa: float
    percent_agreement: float
    macro_f1: float
    weighted_f1: float
    per_code: List[PerCodeMetrics]
    total_segments: int
    grade: str = field(init=False)
    interpretation: str = field(init=False)

    def __post_init__(self) -> None:
        self.grade, self.interpretation = kappa_to_grade(self.cohens_kappa)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohens_kappa": round(self.cohens_kappa, 4),
            "percent_agreement": round(self.percent_agreement, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "grade": self.grade,
            "interpretation": self.interpretation,
            "total_segments": self.total_segments,
            "per_code": [m.to_dict() for m in self.per_code],
        }


@dataclass
class EvalResult:
    """Full evaluation result including metrics and optional narrative."""

    segments: List[CodedSegment]
    metrics: AgreementMetrics
    narrative: Optional[str] = None
    llm_backend: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_backend": self.llm_backend,
            "model": self.model,
            "metrics": self.metrics.to_dict(),
            "narrative": self.narrative,
            "segments": [s.to_dict() for s in self.segments],
        }
