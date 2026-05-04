"""Agreement metric computation for coding-eval."""

from __future__ import annotations

from typing import List, Set, Tuple

from .models import AgreementMetrics, CodedSegment, PerCodeMetrics


def _binary_confusion(
    segments: List[CodedSegment], code: str
) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for a single code across all segments."""
    tp = fp = fn = tn = 0
    for seg in segments:
        h = 1 if code in seg.human_codes else 0
        lm = 1 if (seg.llm_codes is not None and code in seg.llm_codes) else 0
        if h == 1 and lm == 1:
            tp += 1
        elif h == 0 and lm == 1:
            fp += 1
        elif h == 1 and lm == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _per_code_metrics(segments: List[CodedSegment], code: str) -> PerCodeMetrics:
    """Compute binary agreement metrics for a single code."""
    n = len(segments)
    tp, fp, fn, tn = _binary_confusion(segments, code)

    support = tp + fn  # total human-positive for this code

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Per-code Cohen's Kappa (binary)
    po = (tp + tn) / n if n > 0 else 0.0
    pe_pos = ((tp + fp) / n) * ((tp + fn) / n) if n > 0 else 0.0
    pe_neg = ((fn + tn) / n) * ((fp + tn) / n) if n > 0 else 0.0
    pe = pe_pos + pe_neg
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 1e-9 else 1.0

    return PerCodeMetrics(
        code=code,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        cohens_kappa=kappa,
    )


def _all_codes(segments: List[CodedSegment]) -> List[str]:
    """Return sorted list of all codes seen across human and LLM coding."""
    codes: Set[str] = set()
    for seg in segments:
        codes.update(seg.human_codes)
        if seg.llm_codes:
            codes.update(seg.llm_codes)
    return sorted(codes)


def compute_agreement(segments: List[CodedSegment]) -> AgreementMetrics:
    """Compute inter-rater agreement between human and LLM codes.

    Uses per-code binary Cohen's Kappa aggregated by support (weighted) and
    simple macro averaging, alongside exact-set percent agreement and F1 scores.

    Args:
        segments: List of CodedSegment objects. Each segment must have
                  ``human_codes`` set; ``llm_codes`` may be None (treated as
                  empty for agreement purposes).

    Returns:
        AgreementMetrics populated with all agreement statistics.

    Raises:
        ValueError: If ``segments`` is empty.
    """
    if not segments:
        raise ValueError("Cannot compute agreement: no segments provided.")

    codes = _all_codes(segments)
    n = len(segments)

    per_code_results: List[PerCodeMetrics] = []
    f1_scores: List[float] = []
    kappa_scores: List[float] = []
    support_weights: List[int] = []

    for code in codes:
        pcm = _per_code_metrics(segments, code)
        per_code_results.append(pcm)
        f1_scores.append(pcm.f1)
        kappa_scores.append(pcm.cohens_kappa)
        support_weights.append(pcm.support)

    total_support = sum(support_weights)

    if total_support > 0:
        weighted_kappa = (
            sum(k * w for k, w in zip(kappa_scores, support_weights)) / total_support
        )
        weighted_f1 = (
            sum(f * w for f, w in zip(f1_scores, support_weights)) / total_support
        )
    elif kappa_scores:
        weighted_kappa = sum(kappa_scores) / len(kappa_scores)
        weighted_f1 = sum(f1_scores) / len(f1_scores)
    else:
        weighted_kappa = 0.0
        weighted_f1 = 0.0

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Percent exact agreement: exact set match between human and LLM codes
    exact_matches = sum(
        1
        for seg in segments
        if seg.llm_codes is not None
        and set(seg.human_codes) == set(seg.llm_codes)
    )
    percent_agreement = exact_matches / n if n > 0 else 0.0

    return AgreementMetrics(
        cohens_kappa=weighted_kappa,
        percent_agreement=percent_agreement,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_code=per_code_results,
        total_segments=n,
    )
