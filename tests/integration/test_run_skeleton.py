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


from unittest.mock import patch
from pathlib import Path

import pytest


def test_run_pipeline_uses_vision_extract(monkeypatch, tmp_path):
    """After rewire, run_pipeline should call src.pipeline.vision.extract_province
    per province (not the structural extractors).
    """
    from src.pipeline import run as run_mod

    called = []

    async def fake_extract_province(pdf_path, *, province, client, **kwargs):
        called.append((province, Path(pdf_path).name))
        return []

    monkeypatch.setattr(run_mod, "vision_extract_province", fake_extract_province, raising=False)

    # Minimal fixture dirs
    (tmp_path / "pdf").mkdir()
    (tmp_path / "docx").mkdir()
    # Fake PDFs matching the glob patterns so _province_pdf resolves.
    for name in (
        "moh-schedule-benefit-fake.pdf",
        "msc_payment_schedule_-_fake.pdf",
        "yukon_physician_fee_guide_fake.pdf",
    ):
        (tmp_path / "pdf" / name).write_bytes(b"%PDF-fake")

    cfg = run_mod.PipelineConfig(
        raw_pdf_dir=tmp_path / "pdf",
        raw_docx_dir=tmp_path / "docx",
        output_dir=tmp_path / "out",
        diagnostics_dir=tmp_path / "diag",
        version="test",
        force=True,
    )

    with pytest.raises(Exception):
        # Will fail somewhere in NGS/embed with empty records; we only care
        # that vision_extract_province was invoked for each province.
        run_mod.run_pipeline(cfg, client=object())

    assert {c[0] for c in called} == {"ON", "BC", "YT"}
