"""Typer-based CLI. Thin wrapper around src/pipeline/run.py.

Run with:  python -m src.cli run [--version 2026-04-14] [--force]
           python -m src.cli embed [--version 2026-04-14] [--force]
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from src.openai_client import OpenAIClient
from src.pipeline.embed import build_embeddings, save_npz
from src.pipeline.run import PipelineConfig, run_pipeline
from src.pipeline.schema import FeeCodeRecord

app = typer.Typer(help="FSC-NGS pipeline CLI", no_args_is_help=True)


def _configure_logging(verbose: bool = False) -> None:
    """Configure root logger once per CLI invocation."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

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
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="DEBUG-level logging (default: INFO)"
    ),
) -> None:
    """Run the full extraction -> mapping -> embedding pipeline."""
    _configure_logging(verbose=verbose)
    load_dotenv()
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    extract_model = os.environ.get("OPENAI_EXTRACT_MODEL", "gpt-5.4-mini")
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


@app.command()
def embed(
    version: str | None = typer.Option(
        None, help="Target version (default: newest v*/)"
    ),
    force: bool = typer.Option(
        False, "--force", help="Regenerate even if embeddings.npz exists"
    ),
    embed_dim: int | None = typer.Option(
        None, "--embed-dim", help="Embedding dimension (overrides OPENAI_EMBED_DIM)"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="DEBUG-level logging (default: INFO)"
    ),
) -> None:
    """Re-build embeddings from an existing codes.json (no PDFs needed).

    Loads the target version's codes.json and rebuilds embeddings.npz.
    Useful when a collaborator clones the repo where embeddings.npz is gitignored.
    """
    _configure_logging(verbose=verbose)
    load_dotenv()

    logger = logging.getLogger(__name__)

    # Determine the target version directory
    parsed_dir = ROOT / "data" / "parsed"
    if version:
        version_dir = parsed_dir / f"v{version}"
        if not version_dir.exists():
            raise typer.BadParameter(
                f"Version directory not found: {version_dir}"
            )
    else:
        # Find the newest version
        version_dirs = sorted(parsed_dir.glob("v*"))
        if not version_dirs:
            raise typer.BadParameter(
                f"No versioned artifacts in {parsed_dir}"
            )
        version_dir = version_dirs[-1]
        logger.info("Using newest version: %s", version_dir.name)

    codes_path = version_dir / "codes.json"
    embeddings_path = version_dir / "embeddings.npz"

    if not codes_path.exists():
        raise typer.BadParameter(f"codes.json not found at {codes_path}")

    # Skip if embeddings already exist (unless --force)
    if embeddings_path.exists() and not force:
        typer.echo(f"Skipping: embeddings.npz already exists at {embeddings_path} (use --force to redo)")
        return

    # Load records from codes.json
    logger.info("Loading codes from %s", codes_path)
    codes_raw = json.loads(codes_path.read_text(encoding="utf-8"))

    # Schema-version guard
    if codes_raw and codes_raw[0].get("schema_version") != "2":
        raise typer.BadParameter(
            f"Artifact uses schema v{codes_raw[0].get('schema_version')}; "
            "expected v2. Rerun the pipeline: python -m src.cli run --force"
        )

    records = [FeeCodeRecord(**c) for c in codes_raw]
    logger.info("Loaded %d records", len(records))

    # Set up embedding parameters
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    if embed_dim is None:
        embed_dim = int(os.environ.get("OPENAI_EMBED_DIM", "1024"))

    # Build embeddings
    logger.info("Building embeddings (model=%s, dim=%d)", embed_model, embed_dim)
    client = OpenAIClient()
    embeddings, record_ids = build_embeddings(
        records, client=client, model=embed_model, dim=embed_dim
    )

    # Save embeddings
    logger.info("Saving embeddings to %s", embeddings_path)
    save_npz(embeddings_path, embeddings, record_ids)

    # Report costs
    costs = client.costs.snapshot()
    if costs:
        total_tokens = sum(
            c["prompt_tokens"] + c["completion_tokens"]
            for c in costs.values()
        )
        typer.echo(f"Wrote {embeddings_path} ({embeddings.shape[0]} records, {total_tokens} tokens)")
    else:
        typer.echo(f"Wrote {embeddings_path} ({embeddings.shape[0]} records)")


if __name__ == "__main__":
    app()
