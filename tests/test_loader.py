"""Tests for coding_eval.loader."""

import json
import csv
import pytest
from pathlib import Path

from coding_eval.loader import (
    _parse_codes,
    _coerce_codes,
    load_json,
    load_csv,
    load_segments,
    merge_coding_files,
)
from coding_eval.models import CodedSegment

FIXTURE = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# _parse_codes
# ---------------------------------------------------------------------------


class TestParseCodes:
    def test_semicolon_separated(self):
        assert _parse_codes("a;b;c") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert _parse_codes("a ; b ; c") == ["a", "b", "c"]

    def test_empty_string_returns_empty(self):
        assert _parse_codes("") == []

    def test_none_returns_empty(self):
        assert _parse_codes(None) == []  # type: ignore[arg-type]

    def test_single_code(self):
        assert _parse_codes("satisfaction") == ["satisfaction"]

    def test_trailing_semicolon_ignored(self):
        result = _parse_codes("a;b;")
        assert "" not in result

    def test_whitespace_only_returns_empty(self):
        assert _parse_codes("   ") == []


# ---------------------------------------------------------------------------
# _coerce_codes
# ---------------------------------------------------------------------------


class TestCoerceCodes:
    def test_list_passthrough(self):
        assert _coerce_codes(["a", "b"]) == ["a", "b"]

    def test_string_parsed(self):
        assert _coerce_codes("a;b") == ["a", "b"]

    def test_none_returns_empty(self):
        assert _coerce_codes(None) == []

    def test_strips_list_whitespace(self):
        assert _coerce_codes([" a ", " b "]) == ["a", "b"]

    def test_empty_list_returns_empty(self):
        assert _coerce_codes([]) == []


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_loads_human_codes_fixture(self):
        segs = load_json(FIXTURE / "human_codes.json")
        assert len(segs) == 10

    def test_segment_id_string(self):
        segs = load_json(FIXTURE / "human_codes.json")
        for s in segs:
            assert isinstance(s.segment_id, str)

    def test_human_codes_list(self):
        segs = load_json(FIXTURE / "human_codes.json")
        for s in segs:
            assert isinstance(s.human_codes, list)

    def test_llm_codes_none_when_absent(self):
        segs = load_json(FIXTURE / "human_codes.json")
        for s in segs:
            assert s.llm_codes is None

    def test_loads_llm_codes_when_present(self):
        segs = load_json(FIXTURE / "perfect_agreement.json")
        assert all(s.llm_codes is not None for s in segs)

    def test_perfect_fixture_segment_count(self):
        segs = load_json(FIXTURE / "perfect_agreement.json")
        assert len(segs) == 5

    def test_no_agreement_fixture_loads(self):
        segs = load_json(FIXTURE / "no_agreement.json")
        assert len(segs) == 5

    def test_accepts_codes_alias(self, tmp_path):
        data = [{"segment_id": "1", "text": "t", "codes": ["a"]}]
        f = tmp_path / "x.json"
        f.write_text(json.dumps(data))
        segs = load_json(f)
        assert segs[0].human_codes == ["a"]

    def test_accepts_human_codes_alias(self, tmp_path):
        data = [{"segment_id": "1", "text": "t", "human_codes": ["b"]}]
        f = tmp_path / "x.json"
        f.write_text(json.dumps(data))
        segs = load_json(f)
        assert segs[0].human_codes == ["b"]

    def test_empty_codes_field(self, tmp_path):
        data = [{"segment_id": "1", "text": "t", "codes": []}]
        f = tmp_path / "x.json"
        f.write_text(json.dumps(data))
        segs = load_json(f)
        assert segs[0].human_codes == []


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_loads_human_codes_fixture(self):
        segs = load_csv(FIXTURE / "human_codes.csv")
        assert len(segs) == 5

    def test_human_codes_parsed(self):
        segs = load_csv(FIXTURE / "human_codes.csv")
        assert "satisfaction" in segs[0].human_codes

    def test_llm_codes_none_when_absent(self):
        segs = load_csv(FIXTURE / "human_codes.csv")
        for s in segs:
            assert s.llm_codes is None

    def test_loads_llm_codes_column(self):
        segs = load_csv(FIXTURE / "coded_pairs.csv")
        assert all(s.llm_codes is not None for s in segs)

    def test_coded_pairs_count(self):
        segs = load_csv(FIXTURE / "coded_pairs.csv")
        assert len(segs) == 5

    def test_semicolon_parsing(self, tmp_path):
        f = tmp_path / "t.csv"
        f.write_text("segment_id,text,codes\ns1,hello,a;b;c\n")
        segs = load_csv(f)
        assert segs[0].human_codes == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# load_segments
# ---------------------------------------------------------------------------


class TestLoadSegments:
    def test_loads_json_by_extension(self):
        segs = load_segments(str(FIXTURE / "human_codes.json"))
        assert len(segs) == 10

    def test_loads_csv_by_extension(self):
        segs = load_segments(str(FIXTURE / "human_codes.csv"))
        assert len(segs) == 5

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_segments("/nonexistent/path.json")

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_segments(str(f))

    def test_returns_list_of_coded_segments(self):
        segs = load_segments(str(FIXTURE / "human_codes.json"))
        assert all(isinstance(s, CodedSegment) for s in segs)


# ---------------------------------------------------------------------------
# merge_coding_files
# ---------------------------------------------------------------------------


class TestMergeCodingFiles:
    def test_merge_by_segment_id(self):
        segs = merge_coding_files(
            str(FIXTURE / "human_codes.json"),
            str(FIXTURE / "llm_codes_partial.json"),
        )
        assert len(segs) == 10

    def test_llm_codes_populated(self):
        segs = merge_coding_files(
            str(FIXTURE / "human_codes.json"),
            str(FIXTURE / "llm_codes_partial.json"),
        )
        for s in segs:
            assert s.llm_codes is not None

    def test_human_codes_preserved(self):
        segs = merge_coding_files(
            str(FIXTURE / "human_codes.json"),
            str(FIXTURE / "llm_codes_partial.json"),
        )
        human_map = {s.segment_id: s.human_codes for s in segs}
        assert "satisfaction" in human_map["s01"]

    def test_missing_llm_segment_gives_none(self, tmp_path):
        human = [{"segment_id": "s1", "text": "t", "codes": ["a"]},
                 {"segment_id": "s2", "text": "t", "codes": ["b"]}]
        llm = [{"segment_id": "s1", "text": "t", "codes": ["a"]}]
        hf = tmp_path / "h.json"
        lf = tmp_path / "l.json"
        hf.write_text(json.dumps(human))
        lf.write_text(json.dumps(llm))
        segs = merge_coding_files(str(hf), str(lf))
        s2 = next(s for s in segs if s.segment_id == "s2")
        assert s2.llm_codes is None

    def test_merge_returns_coded_segments(self):
        segs = merge_coding_files(
            str(FIXTURE / "human_codes.json"),
            str(FIXTURE / "llm_codes_partial.json"),
        )
        assert all(isinstance(s, CodedSegment) for s in segs)
