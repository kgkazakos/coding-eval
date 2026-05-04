"""LLM coding and narrative generation via DSPy.

This module is reference implementation — not exercised in CI.
Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY and pass
``--llm-backend`` to activate.
"""

from __future__ import annotations

from typing import List, Optional

from .models import CodedSegment

BACKENDS: dict[str, str] = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
    "gemini": "gemini/gemini-2.0-flash",
}


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

try:
    import dspy

    class _CodingSignature(dspy.Signature):
        """Apply thematic codes to a qualitative research segment.

        Given a transcript excerpt and the full coding scheme, return only the
        codes that accurately describe the content. Do not invent new codes.
        """

        segment_text: str = dspy.InputField(
            desc="The transcript or research segment to code"
        )
        available_codes: str = dspy.InputField(
            desc="Semicolon-separated list of available thematic codes"
        )
        codes_applied: str = dspy.OutputField(
            desc=(
                "Semicolon-separated list of codes that apply to this segment. "
                "Use ONLY codes from available_codes. "
                "Return an empty string if no code applies."
            )
        )

    class _NarrativeSignature(dspy.Signature):
        """Interpret coding agreement metrics for a qualitative research audience."""

        metrics_json: str = dspy.InputField(
            desc="JSON summary of inter-rater agreement metrics"
        )
        narrative: str = dspy.OutputField(
            desc=(
                "3–4 sentence plain-language interpretation. Reference Cohen's Kappa "
                "norms (Landis & Koch 1977). Highlight strengths, name codes with "
                "lowest agreement, and recommend whether re-calibration is needed."
            )
        )

    _HAS_DSPY = True

except ImportError:
    _HAS_DSPY = False


# ---------------------------------------------------------------------------
# Public evaluator class
# ---------------------------------------------------------------------------


class LLMCodingEvaluator:
    """Wraps DSPy modules for LLM-based thematic coding and narrative generation.

    Args:
        backend: One of ``"openai"``, ``"anthropic"``, or ``"gemini"``.

    Raises:
        ImportError: If ``dspy-ai`` is not installed.
        ValueError: If an unknown backend is specified.
    """

    def __init__(self, backend: str = "openai") -> None:
        if not _HAS_DSPY:
            raise ImportError(
                "dspy-ai is required for LLM coding. "
                "Install it with: pip install dspy-ai"
            )
        model_str = BACKENDS.get(backend)
        if not model_str:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {list(BACKENDS)}"
            )
        lm = dspy.LM(model_str)  # type: ignore[possibly-undefined]
        dspy.configure(lm=lm)  # type: ignore[possibly-undefined]
        self._coder = dspy.ChainOfThought(_CodingSignature)  # type: ignore[possibly-undefined]
        self._narrator = dspy.ChainOfThought(_NarrativeSignature)  # type: ignore[possibly-undefined]
        self.model = model_str

    def code_segment(self, text: str, available_codes: List[str]) -> List[str]:
        """Code a single segment and return matching codes.

        Args:
            text: The segment text.
            available_codes: Full list of codes in the coding scheme.

        Returns:
            List of codes (subset of ``available_codes``) applied by the LLM.
        """
        result = self._coder(
            segment_text=text,
            available_codes="; ".join(available_codes),
        )
        raw = result.codes_applied or ""
        predicted = [c.strip() for c in raw.split(";") if c.strip()]
        valid = set(available_codes)
        return [c for c in predicted if c in valid]

    def code_segments(
        self,
        segments: List[CodedSegment],
        available_codes: Optional[List[str]] = None,
    ) -> List[CodedSegment]:
        """Re-code all segments and return a new list with ``llm_codes`` set.

        If ``available_codes`` is not supplied the coding scheme is derived
        from the union of all human codes across the segment list.

        Args:
            segments: Human-coded segments to re-code.
            available_codes: Optional explicit coding scheme.

        Returns:
            New list of CodedSegment objects with ``llm_codes`` populated.
        """
        if available_codes is None:
            scheme: set[str] = set()
            for seg in segments:
                scheme.update(seg.human_codes)
            available_codes = sorted(scheme)

        coded: List[CodedSegment] = []
        for seg in segments:
            llm_codes = self.code_segment(seg.text, available_codes)
            coded.append(
                CodedSegment(
                    segment_id=seg.segment_id,
                    text=seg.text,
                    human_codes=seg.human_codes,
                    llm_codes=llm_codes,
                )
            )
        return coded

    def generate_narrative(self, metrics_json: str) -> str:
        """Generate a plain-language interpretation of agreement metrics.

        Args:
            metrics_json: JSON string of AgreementMetrics.to_dict().

        Returns:
            Narrative string.
        """
        result = self._narrator(metrics_json=metrics_json)
        return result.narrative
