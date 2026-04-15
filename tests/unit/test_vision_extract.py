"""Tests for extract_window: prompt assembly + retry-on-ValidationError."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from pydantic import ValidationError

from src.pipeline.vision.extract import extract_window
from src.pipeline.vision.schema import VisionRecord, WindowExtraction
from src.pipeline.vision.toc import SectionContext
from src.pipeline.vision.windows import Window


def _make_window(target_page: int = 2) -> Window:
    return Window(
        target_page=target_page,
        context_page=target_page - 1 if target_page > 1 else None,
        section_hints=SectionContext(chapter="X", section=None, subsection=None),
    )


def _make_client(return_values: list):
    """Build a MagicMock client whose chat_vision_json iterates over values.

    Each entry is either a WindowExtraction instance (returned) or an
    Exception (raised).
    """
    client = MagicMock()
    it = iter(return_values)

    def _side(*args, **kwargs):
        val = next(it)
        if isinstance(val, Exception):
            raise val
        return val

    client.chat_vision_json.side_effect = _side
    return client


def test_extract_window_returns_records():
    client = _make_client([
        WindowExtraction(records=[
            VisionRecord(
                province="ON", fsc_code="A001", fsc_fn="fn", fsc_description="desc",
                page=2, extraction_confidence=0.9,
            ),
        ])
    ])
    window = _make_window(target_page=2)

    records = asyncio.run(
        extract_window(window=window, province="ON", images=[b"img1", b"img2"], client=client)
    )

    assert len(records) == 1
    assert records[0].fsc_code == "A001"
    # First call used temperature=0.0
    assert client.chat_vision_json.call_args_list[0].kwargs["temperature"] == 0.0


def test_extract_window_retries_once_on_validation_error():
    good = WindowExtraction(records=[
        VisionRecord(
            province="ON", fsc_code="A002", fsc_fn="fn", fsc_description="desc",
            page=2, extraction_confidence=0.7,
        ),
    ])

    # Simulate ValidationError by constructing one from a bad payload
    try:
        WindowExtraction(records=[{"bogus": 1}])
    except ValidationError as exc:
        validation_err = exc

    client = _make_client([validation_err, good])
    window = _make_window()

    records = asyncio.run(
        extract_window(window=window, province="ON", images=[b"img"], client=client)
    )

    assert len(records) == 1
    # Retry should have used temperature=0.2
    assert client.chat_vision_json.call_args_list[1].kwargs["temperature"] == 0.2


def test_extract_window_emits_zero_records_on_second_failure():
    try:
        WindowExtraction(records=[{"bogus": 1}])
    except ValidationError as exc:
        validation_err = exc

    client = _make_client([validation_err, validation_err])
    window = _make_window()

    records = asyncio.run(
        extract_window(window=window, province="ON", images=[b"img"], client=client)
    )

    assert records == []
    assert client.chat_vision_json.call_count == 2


def test_extract_window_passes_images_in_context_then_target_order():
    client = _make_client([WindowExtraction(records=[])])
    window = _make_window(target_page=5)
    images = [b"context-img", b"target-img"]

    asyncio.run(
        extract_window(window=window, province="ON", images=images, client=client)
    )

    first_call_images = client.chat_vision_json.call_args.kwargs["images"]
    assert first_call_images == [b"context-img", b"target-img"]


def test_extract_window_writes_failure_log_on_double_fail(tmp_path):
    import json as _json
    try:
        WindowExtraction(records=[{"bogus": 1}])
    except ValidationError as exc:
        validation_err = exc

    client = _make_client([validation_err, validation_err])
    window = _make_window(target_page=42)
    log_path = tmp_path / "diag" / "window_failures.jsonl"

    records = asyncio.run(
        extract_window(
            window=window,
            province="BC",
            images=[b"img"],
            client=client,
            failure_log=log_path,
        )
    )

    assert records == []
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = _json.loads(lines[0])
    assert entry["province"] == "BC"
    assert entry["target_page"] == 42
    assert entry["error_class"] == "ValidationError"
    assert "message" in entry
