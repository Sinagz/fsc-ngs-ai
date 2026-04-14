import json
from pathlib import Path

import pytest

from src.pipeline.run import PipelineConfig, run_pipeline


@pytest.mark.integration
def test_pipeline_skips_completed_steps(tmp_path: Path, monkeypatch):
    # Pre-create the expected final artifact and the pipeline should skip
    version_dir = tmp_path / "parsed" / "v2026-04-13"
    version_dir.mkdir(parents=True)
    (version_dir / "codes.json").write_text("[]", encoding="utf-8")
    (version_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "generated_at": "t",
                "git_sha": "x",
                "row_counts": {"ON": 0, "BC": 0, "YT": 0},
                "source_pdf_hashes": {},
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig(
        raw_pdf_dir=tmp_path / "raw" / "pdf",
        raw_docx_dir=tmp_path / "raw" / "docx",
        output_dir=tmp_path / "parsed",
        diagnostics_dir=tmp_path / "diagnostics",
        version="2026-04-13",
        force=False,
    )
    # Should no-op; force=False and artifacts exist
    result = run_pipeline(cfg, client=None)
    assert result.skipped is True
    # Verify the skip is a true no-op — no side effects
    assert not cfg.diagnostics_dir.exists(), (
        "skip path must not create diagnostics dir"
    )
