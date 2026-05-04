"""coding-eval — Measures LLM thematic coding accuracy against human coders."""

from .agreement import compute_agreement
from .loader import load_segments, merge_coding_files
from .models import AgreementMetrics, CodedSegment, EvalResult, PerCodeMetrics

__all__ = [
    "CodedSegment",
    "AgreementMetrics",
    "EvalResult",
    "PerCodeMetrics",
    "compute_agreement",
    "load_segments",
    "merge_coding_files",
]
