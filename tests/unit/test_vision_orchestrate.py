"""Tests for orchestrate: dedup rule + FeeCodeRecord promotion."""
from __future__ import annotations

from decimal import Decimal

from src.pipeline.vision.orchestrate import _dedup, _to_fee_code_record
from src.pipeline.vision.schema import VisionRecord


def _r(code: str, conf: float, page: int = 1, price: str | None = None) -> VisionRecord:
    return VisionRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="desc",
        page=page, extraction_confidence=conf,
        price=Decimal(price) if price else None,
    )


def test_dedup_keeps_highest_confidence():
    records = [_r("A001", 0.7), _r("A001", 0.95), _r("A001", 0.85)]
    deduped = _dedup(records)
    assert len(deduped) == 1
    assert deduped[0].extraction_confidence == 0.95


def test_dedup_ties_broken_by_lowest_page():
    records = [_r("A001", 0.9, page=3), _r("A001", 0.9, page=1), _r("A001", 0.9, page=2)]
    deduped = _dedup(records)
    assert len(deduped) == 1
    assert deduped[0].page == 1


def test_dedup_preserves_distinct_codes():
    records = [_r("A001", 0.9), _r("A002", 0.9), _r("A003", 0.9)]
    deduped = _dedup(records)
    assert {r.fsc_code for r in deduped} == {"A001", "A002", "A003"}


def test_to_fee_code_record_promotes_fields():
    vr = _r("A001", 0.9, price="25.50")
    fcr = _to_fee_code_record(vr, province="ON", source_pdf_hash="deadbeef")
    assert fcr.schema_version == "2"
    assert fcr.extraction_method == "vision"
    assert fcr.source_pdf_hash == "deadbeef"
    assert fcr.NGS_code is None
    assert fcr.price == Decimal("25.50")
