"""Typer-based CLI. Thin wrapper around src/pipeline/run.py.

Run with:  python -m src.cli run [--version 2026-04-14] [--force]
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from src.openai_client import OpenAIClient
from src.pipeline.run import PipelineConfig, run_pipeline

app = typer.Typer(help="FSC-NGS pipeline CLI", no_args_is_help=True)

ROOT = Path(__file__).resolve().parent.parent


@app.callback()
def _main() -> None:
    """Force typer to treat declared commands as subcommands (e.g. `... run`)."""


def _today() -> str:
    return date.today().isoformat()


@app.command()
def run(
    version: str = typer.Option(
        default_factory=_today,
        help="Output version tag (default: today, YYYY-MM-DD)",
    ),
    force: bool = typer.Option(
        False, "--force", help="Redo steps even if outputs already exist"
    ),
    accept_regression: str | None = typer.Option(
        None,
        "--accept-regression",
        help="Override the regression gate with a justification (stored in manifest)",
    ),
    embed_dim: int = typer.Option(
        None,
        "--embed-dim",
        help="Embedding dimension (overrides OPENAI_EMBED_DIM env var)",
    ),
) -> None:
    """Run the full extraction → mapping → embedding pipeline."""
    load_dotenv()
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    extract_model = os.environ.get("OPENAI_EXTRACT_MODEL", "gpt-4o-mini")
    if embed_dim is None:
        embed_dim = int(os.environ.get("OPENAI_EMBED_DIM", "1024"))

    cfg = PipelineConfig(
        raw_pdf_dir=ROOT / "data" / "raw" / "pdf",
        raw_docx_dir=ROOT / "data" / "raw" / "docx",
        output_dir=ROOT / "data" / "parsed",
        diagnostics_dir=ROOT / "data" / "diagnostics" / version,
        version=version,
        force=force,
        accept_regression=accept_regression,
        embed_model=embed_model,
        embed_dim=embed_dim,
        extract_model=extract_model,
    )
    client = OpenAIClient()
    result = run_pipeline(cfg, client=client)
    if result.skipped:
        typer.echo(f"Skipped: artifacts already exist at {result.version_dir}")
    else:
        assert result.manifest is not None
        typer.echo(
            f"Wrote {result.version_dir} "
            f"(rows: {result.manifest.row_counts})"
        )


if __name__ == "__main__":
    app()
