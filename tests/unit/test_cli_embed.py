"""Test the `python -m src.cli embed` command."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.pipeline.schema import FeeCodeRecord


def test_embed_command_with_explicit_version(tmp_path, monkeypatch):
    """Test embed command reads codes.json from explicit version."""
    # Set up a minimal v* directory with codes.json under data/parsed/
    repo_root = tmp_path
    parsed_dir = repo_root / "data" / "parsed"
    vdir = parsed_dir / "v2026-04-15-test"
    vdir.mkdir(parents=True)

    codes = [
        {
            "schema_version": "2",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "hello world test",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "vision",
            "extraction_confidence": 0.9,
        }
    ]
    (vdir / "codes.json").write_text(json.dumps(codes))

    # Create manifest so the validation passes
    manifest = {
        "schema_version": "1",
        "generated_at": "2026-04-15T10:00:00Z",
        "git_sha": "abc123",
        "row_counts": {"ON": 1, "BC": 0, "YT": 0},
        "source_pdf_hashes": {},
        "models": {"embed": "text-embedding-3-large"},
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    runner = CliRunner()
    with patch("src.cli.build_embeddings") as mock_build, \
         patch("src.cli.save_npz") as mock_save, \
         patch("src.cli.OpenAIClient") as mock_client_cls, \
         patch("src.cli.ROOT", repo_root):
        # Mock the embeddings return
        mock_build.return_value = (
            np.zeros((1, 1024), dtype=np.float32),
            np.array([0], dtype=np.int32),
        )
        mock_client = MagicMock()
        mock_client.costs.snapshot.return_value = {"embed": {"prompt_tokens": 100, "completion_tokens": 0}}
        mock_client_cls.return_value = mock_client

        result = runner.invoke(
            app, ["embed", "--version", "2026-04-15-test"],
            catch_exceptions=False
        )

    assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output:\n{result.output}"
    # Check that build_embeddings was called
    assert mock_build.called
    # Check that save_npz was called
    assert mock_save.called


def test_embed_command_skips_when_npz_exists(tmp_path, monkeypatch):
    """Test embed command skips if embeddings.npz already exists."""
    repo_root = tmp_path
    parsed_dir = repo_root / "data" / "parsed"
    vdir = parsed_dir / "v2026-04-15-test"
    vdir.mkdir(parents=True)

    codes = [
        {
            "schema_version": "2",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "hello world test",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "vision",
            "extraction_confidence": 0.9,
        }
    ]
    (vdir / "codes.json").write_text(json.dumps(codes))
    (vdir / "embeddings.npz").write_text("")  # Dummy file

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    runner = CliRunner()
    with patch("src.cli.build_embeddings") as mock_build, \
         patch("src.cli.OpenAIClient") as mock_client_cls, \
         patch("src.cli.ROOT", repo_root):
        result = runner.invoke(
            app, ["embed", "--version", "2026-04-15-test"]
        )

    assert result.exit_code == 0, f"Expected exit code 0. Output:\n{result.output}"
    # build_embeddings should NOT be called when skipping
    assert not mock_build.called
    assert "skipping" in result.output.lower()


def test_embed_command_force_flag(tmp_path, monkeypatch):
    """Test embed command with --force rebuilds even if npz exists."""
    repo_root = tmp_path
    parsed_dir = repo_root / "data" / "parsed"
    vdir = parsed_dir / "v2026-04-15-test"
    vdir.mkdir(parents=True)

    codes = [
        {
            "schema_version": "2",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "hello world test",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "vision",
            "extraction_confidence": 0.9,
        }
    ]
    (vdir / "codes.json").write_text(json.dumps(codes))
    (vdir / "embeddings.npz").write_text("")  # Dummy file

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    runner = CliRunner()
    with patch("src.cli.build_embeddings") as mock_build, \
         patch("src.cli.save_npz") as mock_save, \
         patch("src.cli.OpenAIClient") as mock_client_cls, \
         patch("src.cli.ROOT", repo_root):
        mock_build.return_value = (
            np.zeros((1, 1024), dtype=np.float32),
            np.array([0], dtype=np.int32),
        )
        mock_client = MagicMock()
        mock_client.costs.snapshot.return_value = {}
        mock_client_cls.return_value = mock_client

        result = runner.invoke(
            app, ["embed", "--version", "2026-04-15-test", "--force"],
            catch_exceptions=False
        )

    assert result.exit_code == 0, f"Expected exit code 0. Output:\n{result.output}"
    # build_embeddings SHOULD be called with --force
    assert mock_build.called


def test_embed_command_missing_version_fails(tmp_path, monkeypatch):
    """Test embed command fails with clear message for non-existent version."""
    repo_root = tmp_path
    repo_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    runner = CliRunner()
    with patch("src.cli.ROOT", repo_root):
        result = runner.invoke(
            app, ["embed", "--version", "nonexistent-version"]
        )

    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


def test_embed_command_finds_newest_version(tmp_path, monkeypatch):
    """Test embed command auto-discovers newest version."""
    # Create multiple versions
    repo_root = tmp_path
    parsed_dir = repo_root / "data" / "parsed"
    vdir_old = parsed_dir / "v2026-04-14"
    vdir_new = parsed_dir / "v2026-04-15"
    vdir_old.mkdir(parents=True)
    vdir_new.mkdir(parents=True)

    codes = [
        {
            "schema_version": "2",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "hello",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "vision",
            "extraction_confidence": 0.9,
        }
    ]
    (vdir_new / "codes.json").write_text(json.dumps(codes))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    runner = CliRunner()
    with patch("src.cli.build_embeddings") as mock_build, \
         patch("src.cli.save_npz") as mock_save, \
         patch("src.cli.OpenAIClient") as mock_client_cls, \
         patch("src.cli.ROOT", repo_root):
        mock_build.return_value = (
            np.zeros((1, 1024), dtype=np.float32),
            np.array([0], dtype=np.int32),
        )
        mock_client = MagicMock()
        mock_client.costs.snapshot.return_value = {}
        mock_client_cls.return_value = mock_client

        result = runner.invoke(
            app, ["embed"],
            catch_exceptions=False
        )

    assert result.exit_code == 0, f"Expected exit code 0. Output:\n{result.output}"
    # Verify it operated on the newest version
    assert "v2026-04-15" in result.output


def test_embed_dim_override(tmp_path, monkeypatch):
    """Test embed command respects --embed-dim flag."""
    repo_root = tmp_path
    parsed_dir = repo_root / "data" / "parsed"
    vdir = parsed_dir / "v2026-04-15-test"
    vdir.mkdir(parents=True)

    codes = [
        {
            "schema_version": "2",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "hello",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "vision",
            "extraction_confidence": 0.9,
        }
    ]
    (vdir / "codes.json").write_text(json.dumps(codes))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_EMBED_DIM", "1024")

    runner = CliRunner()
    with patch("src.cli.build_embeddings") as mock_build, \
         patch("src.cli.save_npz"), \
         patch("src.cli.OpenAIClient") as mock_client_cls, \
         patch("src.cli.load_dotenv"), \
         patch("src.cli.ROOT", repo_root):
        mock_build.return_value = (
            np.zeros((1, 512), dtype=np.float32),
            np.array([0], dtype=np.int32),
        )
        mock_client = MagicMock()
        mock_client.costs.snapshot.return_value = {}
        mock_client_cls.return_value = mock_client

        result = runner.invoke(
            app, ["embed", "--version", "2026-04-15-test", "--embed-dim", "512"],
            catch_exceptions=False
        )

    assert result.exit_code == 0
    # Check that build_embeddings was called with dim=512
    call_kwargs = mock_build.call_args.kwargs
    assert call_kwargs.get("dim") == 512
