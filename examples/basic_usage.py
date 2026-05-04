"""basic_usage.py — coding-eval example without API keys."""

from coding_eval import compute_agreement, CodedSegment

# Simulate a research session: 6 segments, human vs LLM codes
segments = [
    CodedSegment("s1", "The checkout is super smooth", ["satisfaction", "usability"], ["satisfaction", "usability"]),
    CodedSegment("s2", "Search breaks on mobile every time", ["pain_point", "mobile"], ["pain_point"]),
    CodedSegment("s3", "I'd love a wishlist feature", ["feature_request"], ["feature_request"]),
    CodedSegment("s4", "App crashed mid-purchase again", ["bug", "pain_point"], ["bug"]),
    CodedSegment("s5", "Recommendations feel personal", ["satisfaction"], ["satisfaction"]),
    CodedSegment("s6", "Too many taps to check out", ["usability", "pain_point"], ["usability", "pain_point"]),
]

metrics = compute_agreement(segments)

print(f"Cohen's Kappa:      {metrics.cohens_kappa:.3f}")
print(f"Exact agreement:    {metrics.percent_agreement:.1%}")
print(f"Macro F1:           {metrics.macro_f1:.3f}")
print(f"Grade:              {metrics.grade} — {metrics.interpretation}")
print()
print("Per-code breakdown:")
for pc in metrics.per_code:
    print(f"  {pc.code:<20} P={pc.precision:.2f}  R={pc.recall:.2f}  F1={pc.f1:.2f}  κ={pc.cohens_kappa:.2f}")
