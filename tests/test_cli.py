"""Tests for coding_eval CLI commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from coding_eval.cli import cli

FIXTURE = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Help and version
# ---------------------------------------------------------------------------


class TestHelpAndVersion:
    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "coding-eval" in result.output.lower()

    def test_eval_help(self, runner):
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--llm-backend" in result.output

    def test_compare_help(self, runner):
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "HUMAN_FILE" in result.output

    def test_batch_help(self, runner):
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# eval command — no-llm path (llm_codes already in file)
# ---------------------------------------------------------------------------


class TestEvalCommandNoLlm:
    def test_perfect_agreement_renders(self, runner):
        result = runner.invoke(
            cli,
            ["eval", str(FIXTURE / "perfect_agreement.json"), "--llm-backend", "none"],
        )
        assert result.exit_code == 0

    def test_perfect_agreement_json_output(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "metrics" in data

    def test_json_metrics_structure(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        metrics = data["metrics"]
        assert "cohens_kappa" in metrics
        assert "percent_agreement" in metrics
        assert "grade" in metrics
        assert "per_code" in metrics

    def test_perfect_kappa_is_one(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        assert abs(data["metrics"]["cohens_kappa"] - 1.0) < 1e-4

    def test_perfect_grade_is_a(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        assert data["metrics"]["grade"] == "A"

    def test_no_agreement_json_output(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "no_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["metrics"]["percent_agreement"] == 0.0

    def test_missing_llm_codes_exits_nonzero(self, runner):
        # human_codes.json has no llm_codes field
        result = runner.invoke(
            cli,
            ["eval", str(FIXTURE / "human_codes.json"), "--llm-backend", "none"],
        )
        assert result.exit_code != 0

    def test_json_output_has_segments(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        assert isinstance(data["segments"], list)
        assert len(data["segments"]) == 5

    def test_json_output_segments_have_required_fields(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        for seg in data["segments"]:
            assert "segment_id" in seg
            assert "human_codes" in seg
            assert "llm_codes" in seg

    def test_no_narrative_flag(self, runner):
        result = runner.invoke(
            cli,
            [
                "eval",
                str(FIXTURE / "perfect_agreement.json"),
                "--llm-backend", "none",
                "--no-narrative",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["narrative"] is None


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------


class TestCompareCommand:
    def test_compare_basic(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                str(FIXTURE / "human_codes.json"),
                str(FIXTURE / "llm_codes_partial.json"),
                "--no-narrative",
            ],
        )
        assert result.exit_code == 0

    def test_compare_json_output(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                str(FIXTURE / "human_codes.json"),
                str(FIXTURE / "llm_codes_partial.json"),
                "--no-narrative",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "metrics" in data

    def test_compare_metrics_structure(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                str(FIXTURE / "human_codes.json"),
                str(FIXTURE / "llm_codes_partial.json"),
                "--no-narrative",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        m = data["metrics"]
        for key in ("cohens_kappa", "percent_agreement", "macro_f1", "grade"):
            assert key in m

    def test_compare_kappa_in_range(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                str(FIXTURE / "human_codes.json"),
                str(FIXTURE / "llm_codes_partial.json"),
                "--no-narrative",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        assert -1.0 <= data["metrics"]["cohens_kappa"] <= 1.0

    def test_compare_per_code_list(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                str(FIXTURE / "human_codes.json"),
                str(FIXTURE / "llm_codes_partial.json"),
                "--no-narrative",
                "--json-output",
            ],
        )
        data = json.loads(result.output)
        assert isinstance(data["metrics"]["per_code"], list)
        assert len(data["metrics"]["per_code"]) > 0

    def test_compare_csv_files(self, runner):
        result = runner.invoke(
            cli,
            [
                "compare",
                str(FIXTURE / "human_codes.csv"),
                str(FIXTURE / "human_codes.csv"),
                "--no-narrative",
                "--json-output",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Comparing file to itself: perfect agreement
        assert data["metrics"]["percent_agreement"] == 1.0

    def test_compare_nonexistent_file_fails(self, runner):
        result = runner.invoke(
            cli,
            ["compare", "/nonexistent.json", str(FIXTURE / "human_codes.json"), "--no-narrative"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


class TestBatchCommand:
    def test_batch_empty_dir(self, runner, tmp_path):
        result = runner.invoke(cli, ["batch", str(tmp_path)])
        assert result.exit_code == 0  # No-op, not an error

    def test_batch_with_pre_coded_files(self, runner, tmp_path):
        import json as _json
        data = [
            {"segment_id": "s1", "text": "t", "codes": ["a"], "llm_codes": ["a"]},
            {"segment_id": "s2", "text": "t", "codes": ["b"], "llm_codes": ["b"]},
        ]
        (tmp_path / "test.json").write_text(_json.dumps(data))
        result = runner.invoke(cli, ["batch", str(tmp_path), "--json-output"])
        assert result.exit_code == 0
        results = json.loads(result.output)
        assert len(results) == 1
        assert results[0]["file"] == "test.json"

    def test_batch_json_output_structure(self, runner, tmp_path):
        import json as _json
        data = [
            {"segment_id": "s1", "text": "t", "codes": ["a"], "llm_codes": ["a"]},
        ]
        (tmp_path / "f.json").write_text(_json.dumps(data))
        result = runner.invoke(cli, ["batch", str(tmp_path), "--json-output"])
        results = json.loads(result.output)
        assert "metrics" in results[0]

    def test_batch_skips_files_without_llm_codes(self, runner, tmp_path):
        import json as _json
        data = [{"segment_id": "s1", "text": "t", "codes": ["a"]}]
        (tmp_path / "nollm.json").write_text(_json.dumps(data))
        result = runner.invoke(cli, ["batch", str(tmp_path), "--json-output"])
        assert result.exit_code == 0
        results = json.loads(result.output)
        assert results == []

    def test_batch_custom_pattern(self, runner, tmp_path):
        import json as _json
        data = [{"segment_id": "s1", "text": "t", "codes": ["a"], "llm_codes": ["a"]}]
        (tmp_path / "data.json").write_text(_json.dumps(data))
        result = runner.invoke(
            cli, ["batch", str(tmp_path), "--pattern", "*.json", "--json-output"]
        )
        assert result.exit_code == 0
        results = json.loads(result.output)
        assert len(results) == 1
