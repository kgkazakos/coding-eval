"""CLI for coding-eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agreement import compute_agreement
from .loader import load_segments, merge_coding_files
from .models import EvalResult

console = Console()

_GRADE_COLOR = {
    "A": "bright_green",
    "B": "cyan",
    "C": "yellow",
    "D": "orange1",
    "F": "red",
}

_BACKENDS = ["openai", "anthropic", "gemini", "none"]


def _kappa_color(k: float) -> str:
    if k >= 0.80:
        return "bright_green"
    if k >= 0.60:
        return "cyan"
    if k >= 0.40:
        return "yellow"
    if k >= 0.20:
        return "orange1"
    return "red"


def _render(result: EvalResult) -> None:
    m = result.metrics
    gc = _GRADE_COLOR.get(m.grade, "white")

    backend_line = ""
    if result.llm_backend and result.llm_backend != "none":
        backend_line = (
            f"\nBackend: [dim]{result.llm_backend}[/] · [dim]{result.model}[/]"
        )

    console.print()
    console.print(
        Panel.fit(
            f"[bold {gc}]{m.grade}[/]  {m.interpretation}\n"
            f"κ = [bold]{m.cohens_kappa:.3f}[/]  ·  "
            f"Exact agreement = [bold]{m.percent_agreement:.1%}[/]  ·  "
            f"Macro F1 = [bold]{m.macro_f1:.3f}[/]  ·  "
            f"Weighted F1 = [bold]{m.weighted_f1:.3f}[/]\n"
            f"Segments evaluated: [bold]{m.total_segments}[/]"
            + backend_line,
            title="[bold]Coding Agreement[/]",
            border_style="dim",
            padding=(0, 2),
        )
    )

    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        header_style="bold dim",
        padding=(0, 1),
    )
    table.add_column("Code", style="bold", no_wrap=True)
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("κ", justify="right")
    table.add_column("Support", justify="right")

    for pc in m.per_code:
        kc = _kappa_color(pc.cohens_kappa)
        table.add_row(
            pc.code,
            f"{pc.precision:.3f}",
            f"{pc.recall:.3f}",
            f"{pc.f1:.3f}",
            f"[{kc}]{pc.cohens_kappa:.3f}[/]",
            str(pc.support),
        )

    console.print(table)

    if result.narrative:
        console.print(
            Panel(
                result.narrative,
                title="[bold]Narrative[/]",
                border_style="dim",
                padding=(0, 2),
            )
        )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="coding-eval")
def cli() -> None:
    """coding-eval — Measures LLM thematic coding accuracy against human coders."""


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------


@cli.command("eval")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--llm-backend",
    default="openai",
    type=click.Choice(_BACKENDS),
    show_default=True,
    help=(
        "LLM backend for re-coding segments. "
        "Defaults to openai. "
        "Pass 'none' only if the input file already contains llm_codes."
    ),
)
@click.option(
    "--scheme",
    type=click.Path(exists=True),
    default=None,
    help="JSON file with list of available codes (overrides codes derived from data).",
)
@click.option("--no-narrative", is_flag=True, help="Skip narrative generation.")
@click.option("--json-output", is_flag=True, help="Emit JSON instead of rich display.")
def eval_cmd(
    file: Path,
    llm_backend: str,
    scheme: Optional[str],
    no_narrative: bool,
    json_output: bool,
) -> None:
    """Re-code FILE with LLM and compute agreement against human codes.

    FILE must be .json or .csv with fields: segment_id, text, codes.
    """
    try:
        segments = load_segments(str(file))
    except Exception as exc:
        console.print(f"[red]Failed to load file: {exc}[/]")
        sys.exit(1)

    if not segments:
        console.print("[red]No segments found in file.[/]")
        sys.exit(1)

    available_codes: Optional[List[str]] = None
    if scheme:
        with open(scheme, encoding="utf-8") as fh:
            available_codes = json.load(fh)

    # ------------------------------------------------------------------
    # Rule-based path: llm_backend=none — requires llm_codes in file
    # ------------------------------------------------------------------
    if llm_backend == "none":
        missing = [s for s in segments if s.llm_codes is None]
        if missing:
            console.print(
                f"[red]{len(missing)} segment(s) have no llm_codes. "
                "Use --llm-backend to run LLM coding, or provide a file that "
                "already contains an llm_codes field.[/]"
            )
            sys.exit(1)

        metrics = compute_agreement(segments)
        result = EvalResult(segments=segments, metrics=metrics, llm_backend=None)

        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            _render(result)
        return

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------
    try:
        from .coder import LLMCodingEvaluator
    except ImportError as exc:
        console.print(f"[red]LLM backend unavailable: {exc}[/]")
        sys.exit(1)

    try:
        evaluator = LLMCodingEvaluator(backend=llm_backend)
    except Exception as exc:
        console.print(f"[red]Could not initialise {llm_backend} backend: {exc}[/]")
        sys.exit(1)

    with console.status(
        f"[dim]Coding {len(segments)} segments with {llm_backend}…[/]"
    ):
        coded_segments = evaluator.code_segments(segments, available_codes)

    metrics = compute_agreement(coded_segments)

    narrative: Optional[str] = None
    if not no_narrative:
        with console.status("[dim]Generating narrative…[/]"):
            try:
                narrative = evaluator.generate_narrative(
                    json.dumps(metrics.to_dict(), indent=2)
                )
            except Exception:
                narrative = None

    result = EvalResult(
        segments=coded_segments,
        metrics=metrics,
        narrative=narrative,
        llm_backend=llm_backend,
        model=evaluator.model,
    )

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------


@cli.command("compare")
@click.argument("human_file", type=click.Path(exists=True))
@click.argument("llm_file", type=click.Path(exists=True))
@click.option(
    "--llm-backend",
    default="openai",
    type=click.Choice(_BACKENDS),
    show_default=True,
    help="LLM backend for narrative generation only.",
)
@click.option("--no-narrative", is_flag=True, help="Skip narrative generation.")
@click.option("--json-output", is_flag=True, help="Emit JSON instead of rich display.")
def compare_cmd(
    human_file: str,
    llm_file: str,
    llm_backend: str,
    no_narrative: bool,
    json_output: bool,
) -> None:
    """Compare HUMAN_FILE against LLM_FILE and compute agreement metrics.

    Both files must be .json or .csv with fields: segment_id, text, codes.
    Segments are matched by segment_id.
    """
    try:
        segments = merge_coding_files(human_file, llm_file)
    except Exception as exc:
        console.print(f"[red]Failed to merge files: {exc}[/]")
        sys.exit(1)

    if not segments:
        console.print("[red]No segments found after merging.[/]")
        sys.exit(1)

    metrics = compute_agreement(segments)

    narrative: Optional[str] = None
    model: Optional[str] = None

    if not no_narrative and llm_backend != "none":
        try:
            from .coder import LLMCodingEvaluator

            evaluator = LLMCodingEvaluator(backend=llm_backend)
            model = evaluator.model
            with console.status("[dim]Generating narrative…[/]"):
                narrative = evaluator.generate_narrative(
                    json.dumps(metrics.to_dict(), indent=2)
                )
        except Exception:
            narrative = None

    result = EvalResult(
        segments=segments,
        metrics=metrics,
        narrative=narrative,
        llm_backend=llm_backend if not no_narrative else None,
        model=model,
    )

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


@cli.command("batch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--pattern", default="*.json", show_default=True, help="Glob pattern.")
@click.option("--json-output", is_flag=True, help="Emit JSON array.")
def batch_cmd(directory: str, pattern: str, json_output: bool) -> None:
    """Compute agreement for all pre-coded files in DIRECTORY.

    Files must already contain both a ``codes``/``human_codes`` field and an
    ``llm_codes`` field per segment.
    """
    dir_path = Path(directory)
    files = sorted(dir_path.glob(pattern))

    if not files:
        console.print(
            f"[yellow]No files matching '{pattern}' in {directory}.[/]"
        )
        sys.exit(0)

    results = []
    for fpath in files:
        try:
            segs = load_segments(str(fpath))
            segs_with_llm = [s for s in segs if s.llm_codes is not None]
            if not segs_with_llm:
                if not json_output:
                    console.print(
                        f"[yellow]Skipping {fpath.name}: no llm_codes found[/]"
                    )
                continue
            metrics = compute_agreement(segs_with_llm)
            entry = {"file": fpath.name, "metrics": metrics.to_dict()}
            results.append(entry)
            if not json_output:
                gc = _GRADE_COLOR.get(metrics.grade, "white")
                console.print(
                    f"[bold]{fpath.name}[/]  "
                    f"κ={metrics.cohens_kappa:.3f}  "
                    f"[{gc}][{metrics.grade}][/]  "
                    f"{metrics.interpretation}"
                )
        except Exception as exc:
            if not json_output:
                console.print(f"[red]Error processing {fpath.name}: {exc}[/]")

    if json_output:
        click.echo(json.dumps(results, indent=2))
