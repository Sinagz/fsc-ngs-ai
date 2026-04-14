from decimal import Decimal

from src.pipeline.schema import CandidateRow
from src.pipeline.validate import validate, ValidationError


def _row(province="ON", code="K040", desc="desc", price="78.45", conf=1.0):
    return CandidateRow(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description=desc,
        page=1, source_pdf_hash="a" * 64, confidence=conf,
        price=Decimal(price) if price else None,
    )


def test_validate_accepts_clean_row():
    rows, rejects = validate([_row()])
    assert len(rows) == 1
    assert rejects == []
    assert rows[0].fsc_code == "K040"


def test_validate_rejects_wrong_province_regex():
    rows, rejects = validate([_row(province="ON", code="01712")])
    assert rows == []
    assert len(rejects) == 1
    assert "regex" in rejects[0].reason.lower()


def test_validate_rejects_duplicate_within_province():
    rows, rejects = validate([_row(), _row()])
    assert len(rows) == 1
    assert len(rejects) == 1
    assert "duplicate" in rejects[0].reason.lower()


def test_validate_allows_same_code_across_provinces():
    rows, rejects = validate([_row(province="ON", code="K040"),
                              _row(province="BC", code="01712")])
    assert len(rows) == 2
    assert rejects == []
