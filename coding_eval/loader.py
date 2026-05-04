"""File loading utilities for coding-eval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

from .models import CodedSegment


def _parse_codes(raw: str) -> List[str]:
    """Parse a semicolon-delimited code string into a list."""
    if not raw or not raw.strip():
        return []
    return [c.strip() for c in raw.split(";") if c.strip()]


def _coerce_codes(value: object) -> List[str]:
    """Accept either a list or a semicolon-string and return List[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(c).strip() for c in value if str(c).strip()]
    return _parse_codes(str(value))


def load_json(path: Path) -> List[CodedSegment]:
    """Load a JSON coding file.

    Expected schema per item::

        {
            "segment_id": "s1",
            "text": "...",
            "codes": ["theme_a", "theme_b"],   # or "human_codes"
            "llm_codes": ["theme_a"]            # optional
        }
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    segments: List[CodedSegment] = []
    for item in data:
        raw_codes = item.get("codes") or item.get("human_codes") or []
        raw_llm = item.get("llm_codes")

        segments.append(
            CodedSegment(
                segment_id=str(item["segment_id"]),
                text=str(item["text"]),
                human_codes=_coerce_codes(raw_codes),
                llm_codes=_coerce_codes(raw_llm) if raw_llm is not None else None,
            )
        )
    return segments


def load_csv(path: Path) -> List[CodedSegment]:
    """Load a CSV coding file.

    Expected columns: ``segment_id``, ``text``, ``codes`` (or ``human_codes``).
    Optional column: ``llm_codes``.
    Codes within a cell are semicolon-separated.
    """
    segments: List[CodedSegment] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_codes = row.get("codes") or row.get("human_codes") or ""
            raw_llm: Optional[str] = row.get("llm_codes")

            segments.append(
                CodedSegment(
                    segment_id=str(row["segment_id"]),
                    text=str(row["text"]),
                    human_codes=_parse_codes(raw_codes),
                    llm_codes=_parse_codes(raw_llm) if raw_llm else None,
                )
            )
    return segments


def load_segments(path: str) -> List[CodedSegment]:
    """Load segments from a JSON or CSV file.

    Args:
        path: Path to the coding file.

    Returns:
        List of CodedSegment objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not ``.json`` or ``.csv``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        return load_json(p)
    if suffix == ".csv":
        return load_csv(p)
    raise ValueError(
        f"Unsupported file format: '{suffix}'. Use .json or .csv"
    )


def merge_coding_files(human_path: str, llm_path: str) -> List[CodedSegment]:
    """Merge a human-coding file and an LLM-coding file by segment_id.

    The LLM file's ``codes`` / ``human_codes`` field is treated as the
    predicted codes and stored in ``llm_codes`` on the merged segment.
    Segments present in ``human_path`` but absent in ``llm_path`` will have
    ``llm_codes=None``.

    Args:
        human_path: Path to the human-coded file.
        llm_path: Path to the LLM-coded file.

    Returns:
        Merged list of CodedSegment objects.
    """
    human_segs = {s.segment_id: s for s in load_segments(human_path)}
    llm_segs = {s.segment_id: s for s in load_segments(llm_path)}

    merged: List[CodedSegment] = []
    for sid, human_seg in human_segs.items():
        llm_seg = llm_segs.get(sid)
        llm_codes: Optional[List[str]] = (
            llm_seg.human_codes if llm_seg is not None else None
        )
        merged.append(
            CodedSegment(
                segment_id=sid,
                text=human_seg.text,
                human_codes=human_seg.human_codes,
                llm_codes=llm_codes,
            )
        )
    return merged
