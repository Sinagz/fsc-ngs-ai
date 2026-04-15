from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.pipeline.vision.schema import VisionRecord, WindowExtraction


def test_vision_record_minimal():
    r = VisionRecord(
        province="ON",
        fsc_code="A001",
        fsc_fn="consult",
        fsc_description="Initial consultation",
        page=7,
        extraction_confidence=0.9,
    )
    assert r.price is None
    assert r.fsc_chapter is None


def test_vision_record_with_price_and_hints():
    r = VisionRecord(
        province="BC",
        fsc_code="00025",
        fsc_fn="chamber 1st hour",
        fsc_description="Where no other fee is charged - physician in chamber",
        fsc_chapter="Hyperbaric Chamber",
        fsc_section="Fees",
        price=Decimal("83.68"),
        page=60,
        extraction_confidence=0.95,
    )
    assert r.price == Decimal("83.68")
    assert r.fsc_chapter == "Hyperbaric Chamber"


def test_vision_record_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        VisionRecord(
            province="ON",
            fsc_code="A001",
            fsc_fn="x",
            fsc_description="y",
            page=1,
            extraction_confidence=1.5,
        )


def test_vision_record_rejects_extra_fields():
    with pytest.raises(ValidationError):
        VisionRecord(
            province="ON",
            fsc_code="A001",
            fsc_fn="x",
            fsc_description="y",
            page=1,
            extraction_confidence=0.5,
            bogus="nope",
        )


def test_window_extraction_empty():
    w = WindowExtraction(records=[])
    assert w.records == []


def test_window_extraction_multi_record():
    r1 = VisionRecord(
        province="ON",
        fsc_code="A001",
        fsc_fn="x",
        fsc_description="y",
        page=1,
        extraction_confidence=0.9,
    )
    r2 = VisionRecord(
        province="ON",
        fsc_code="A002",
        fsc_fn="x",
        fsc_description="y",
        page=1,
        extraction_confidence=0.8,
    )
    w = WindowExtraction(records=[r1, r2])
    assert len(w.records) == 2
