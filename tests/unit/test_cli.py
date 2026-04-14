from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from src.cli import app
from src.pipeline.run import PipelineResult


def test_run_invokes_pipeline_with_skip_message():
    runner = CliRunner()
    with patch("src.cli.run_pipeline") as fake_run, \
         patch("src.cli.OpenAIClient") as fake_client:
        fake_run.return_value = PipelineResult(
            skipped=True, version_dir=Path("/tmp/v2026-04-14"), manifest=None,
        )
        fake_client.return_value = "CLIENT"
        result = runner.invoke(app, ["run", "--version", "2026-04-14"])

    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output
    fake_run.assert_called_once()
    cfg = fake_run.call_args.args[0]
    assert cfg.version == "2026-04-14"
    assert cfg.force is False


def test_run_passes_force_flag():
    runner = CliRunner()
    with patch("src.cli.run_pipeline") as fake_run, \
         patch("src.cli.OpenAIClient"):
        fake_run.return_value = PipelineResult(
            skipped=True, version_dir=Path("/tmp/x"), manifest=None,
        )
        result = runner.invoke(app, ["run", "--version", "2026-04-14", "--force"])
    assert result.exit_code == 0, result.output
    cfg = fake_run.call_args.args[0]
    assert cfg.force is True


def test_embed_dim_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_EMBED_DIM", "3072")
    runner = CliRunner()
    with patch("src.cli.run_pipeline") as fake_run, \
         patch("src.cli.OpenAIClient"), \
         patch("src.cli.load_dotenv"):  # don't re-read .env over our env
        fake_run.return_value = PipelineResult(
            skipped=True, version_dir=Path("/tmp/x"), manifest=None,
        )
        result = runner.invoke(app, ["run", "--version", "2026-04-14"])
    assert result.exit_code == 0, result.output
    cfg = fake_run.call_args.args[0]
    assert cfg.embed_dim == 3072


def test_embed_dim_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_EMBED_DIM", "3072")
    runner = CliRunner()
    with patch("src.cli.run_pipeline") as fake_run, \
         patch("src.cli.OpenAIClient"), \
         patch("src.cli.load_dotenv"):
        fake_run.return_value = PipelineResult(
            skipped=True, version_dir=Path("/tmp/x"), manifest=None,
        )
        result = runner.invoke(
            app, ["run", "--version", "2026-04-14", "--embed-dim", "512"]
        )
    assert result.exit_code == 0, result.output
    cfg = fake_run.call_args.args[0]
    assert cfg.embed_dim == 512
