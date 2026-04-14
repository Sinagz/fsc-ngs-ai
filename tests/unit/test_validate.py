from decimal import Decimal

from src.pipeline.schema import CandidateRow
from src.pipeline.validate import validate, ValidationReject


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


def test_validate_rejects_zero_price():
    rows, rejects = validate([_row(price="0")])
    assert rows == []
    assert "positive" in rejects[0].reason.lower()


def test_validate_rejects_negative_price():
    rows, rejects = validate([_row(price="-5.00")])
    assert rows == []
    assert "positive" in rejects[0].reason.lower()


def test_validate_accepts_none_price():
    rows, rejects = validate([_row(price=None)])
    assert len(rows) == 1
    assert rows[0].price is None
    assert rejects == []


def test_validate_sets_extraction_method_from_origin():
    """A rescued row (origin='semantic') should carry that label through,
    regardless of its confidence value (regression guard for Task 10 fix)."""
    row = CandidateRow(
        province="ON", fsc_code="K040", fsc_fn="fn", fsc_description="d",
        price=Decimal("10.00"), page=1, source_pdf_hash="a" * 64,
        confidence=0.95,           # high confidence...
        origin="semantic",         # ...but LLM-rescued
    )
    rows, _ = validate([row])
    assert rows[0].extraction_method == "semantic"


def test_validate_ghost_duplicate_bug_fixed():
    """Regression test: a rejected row (bad price) must NOT occupy the
    `seen` slot and block a later valid row with the same code."""
    bad = _row(price="0")
    good = _row(price="78.45")
    rows, rejects = validate([bad, good])
    assert len(rows) == 1  # good should pass
    assert rows[0].price == Decimal("78.45")
    # Exactly one reject (the bad-price row), not two
    assert len(rejects) == 1
    assert "positive" in rejects[0].reason.lower()
