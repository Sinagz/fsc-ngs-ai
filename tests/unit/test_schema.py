from decimal import Decimal
import pytest
from pydantic import ValidationError

from src.pipeline.schema import (
    FeeCodeRecord, NGSRecord, Manifest, CandidateRow, PageBlock,
)


def _valid_record_kwargs() -> dict:
    return dict(
        province="ON", fsc_code="K040", fsc_fn="Periodic health visit",
        fsc_description="General periodic visit, adult.", page=47,
        source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
    )


def test_record_roundtrips_through_json():
    r = FeeCodeRecord(**_valid_record_kwargs(), price=Decimal("78.45"))
    data = r.model_dump(mode="json")
    assert FeeCodeRecord(**data) == r


def test_record_is_frozen():
    r = FeeCodeRecord(**_valid_record_kwargs())
    with pytest.raises(ValidationError):
        r.province = "BC"  # type: ignore[misc]


def test_record_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        FeeCodeRecord(**_valid_record_kwargs(), nope="x")


def test_record_rejects_unknown_province():
    kwargs = _valid_record_kwargs() | {"province": "AB"}
    with pytest.raises(ValidationError):
        FeeCodeRecord(**kwargs)


def test_record_rejects_negative_confidence():
    kwargs = _valid_record_kwargs() | {"extraction_confidence": -0.1}
    with pytest.raises(ValidationError):
        FeeCodeRecord(**kwargs)


def test_candidate_row_is_frozen():
    row = CandidateRow(
        province="ON", fsc_code="K040", fsc_fn="", fsc_description="",
        page=1, confidence=0.9, source_pdf_hash="a" * 64,
    )
    with pytest.raises(ValidationError):
        row.confidence = 0.1  # type: ignore[misc]


def test_manifest_records_schema_version():
    m = Manifest(
        schema_version="1", generated_at="2026-04-13T00:00:00Z",
        git_sha="deadbeef", row_counts={"ON": 0, "BC": 0, "YT": 0},
        source_pdf_hashes={}, models={"embed": "text-embedding-3-large"},
    )
    assert m.schema_version == "1"


def test_ngs_record_roundtrips():
    n = NGSRecord(ngs_code="1AA", ngs_label="Health exam",
                  ngs_description="routine visit", code_refs=["K040"])
    assert NGSRecord(**n.model_dump()) == n


def test_extraction_method_accepts_vision():
    record = FeeCodeRecord(
        province="ON",
        fsc_code="A001",
        fsc_fn="test",
        fsc_description="test description",
        page=1,
        source_pdf_hash="abc",
        extraction_method="vision",
        extraction_confidence=0.95,
    )
    assert record.extraction_method == "vision"
