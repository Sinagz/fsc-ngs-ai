"""End-to-end integration test against a 4-page mini PDF, using a
pre-recorded hishel cache so CI runs offline without OPENAI_API_KEY."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from src.openai_client import OpenAIClient
from src.pipeline.vision import extract_province

FIXTURE_PDF = Path("tests/fixtures/mini_pdf.pdf")
FIXTURE_CACHE_DIR = Path("tests/fixtures/vision_cache_dir")


@pytest.mark.integration
def test_vision_pipeline_end_to_end_from_cache(tmp_path, monkeypatch):
    # Copy cache dir to tmp so the test doesn't mutate the fixture
    cache_copy = tmp_path / "cache"
    shutil.copytree(FIXTURE_CACHE_DIR, cache_copy)

    # Sentinel API key — all calls should hit cache, so no auth is needed.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-replay-only")
    client = OpenAIClient(cache_dir=cache_copy)
    records = asyncio.run(
        extract_province(FIXTURE_PDF, province="ON", client=client, concurrency=1)
    )

    assert len(records) >= 1
    for r in records:
        assert r.extraction_method == "vision"
        assert r.schema_version == "2"
        assert r.source_pdf_hash
