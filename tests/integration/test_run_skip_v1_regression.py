"""When the previous artifact is schema v1, Phase 5 regression diff
should log a warning and skip instead of crashing with a pydantic error.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_run_skips_regression_when_prior_is_v1(monkeypatch, tmp_path, caplog):
    from src.pipeline import run as run_mod
    from src.pipeline.schema import FeeCodeRecord

    # Fixture: prior v1 directory with one v1 record
    prior_dir = tmp_path / "out" / "v2025-01-01"
    prior_dir.mkdir(parents=True)
    (prior_dir / "codes.json").write_text(json.dumps([
        {
            "schema_version": "1",
            "province": "ON", "fsc_code": "A001", "fsc_fn": "fn",
            "fsc_description": "desc", "page": 1, "source_pdf_hash": "h",
            "extraction_method": "structural", "extraction_confidence": 0.9,
        }
    ]))

    # Minimal PDF dir
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    (pdf_dir / "moh-schedule-benefit-fake.pdf").write_bytes(b"%PDF-fake")
    (pdf_dir / "msc_payment_schedule_-_fake.pdf").write_bytes(b"%PDF-fake")
    (pdf_dir / "yukon_physician_fee_guide_fake.pdf").write_bytes(b"%PDF-fake")

    docx_dir = tmp_path / "docx"
    docx_dir.mkdir()

    async def fake_extract_province(pdf_path, *, province, client, **kwargs):
        return [FeeCodeRecord(
            province=province, fsc_code=f"{province}001", fsc_fn="fn",
            fsc_description="desc", page=1, source_pdf_hash="new",
            extraction_method="vision", extraction_confidence=0.9,
        )]

    monkeypatch.setattr(run_mod, "vision_extract_province", fake_extract_province)
    monkeypatch.setattr(run_mod, "map_ngs", lambda records, ngs_records, **kw: records)
    monkeypatch.setattr(run_mod, "parse_ngs_docx", lambda p: [])
    monkeypatch.setattr(run_mod, "build_embeddings", lambda records, **kw: (
        __import__("numpy").zeros((len(records), 1024), dtype=__import__("numpy").float32),
        __import__("numpy").arange(len(records), dtype=__import__("numpy").int32),
    ))

    cfg = run_mod.PipelineConfig(
        raw_pdf_dir=pdf_dir,
        raw_docx_dir=docx_dir,
        output_dir=tmp_path / "out",
        diagnostics_dir=tmp_path / "diag",
        version="2026-04-15-test",
        force=True,
    )

    client = MagicMock()
    client.costs.snapshot.return_value = {}

    import logging
    caplog.set_level(logging.INFO, logger="src.pipeline.run")

    # Should not raise
    result = run_mod.run_pipeline(cfg, client=client)
    assert not result.skipped

    # Should have logged the v1-skip note
    messages = [r.message for r in caplog.records]
    assert any("schema v1" in m and "skipping" in m.lower() for m in messages)
