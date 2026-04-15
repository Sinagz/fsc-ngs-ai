"""Tests for orchestrate: sanitize + dedup rules + FeeCodeRecord promotion."""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.pipeline.vision.orchestrate import _dedup, _sanitize, _to_fee_code_record
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


# ---------------------------------------------------------------------------
# _sanitize tests
# ---------------------------------------------------------------------------

class TestSanitize:
    """All five sanitization rules."""

    def test_strip_hash_prefix(self):
        """Leading '#' (with optional space) is stripped."""
        result = _sanitize([_r("# E190", 0.9), _r("#E191", 0.85)])
        codes = [r.fsc_code for r in result]
        assert codes == ["E190", "E191"]

    def test_strip_star_prefix(self):
        """Leading '*' (with optional space) is stripped."""
        result = _sanitize([_r("*51016", 0.9), _r("* 2017", 0.8)])
        codes = [r.fsc_code for r in result]
        assert codes == ["51016", "2017"]

    def test_strip_internal_whitespace(self):
        """Internal whitespace is removed so 'E 190' becomes 'E190'."""
        result = _sanitize([_r("E 190", 0.9), _r("B 0010", 0.8)])
        codes = [r.fsc_code for r in result]
        assert codes == ["E190", "B0010"]

    def test_split_comma_separated_codes(self):
        """A comma-separated code string expands into multiple records."""
        original = _r("01088, 01090, 01091", 0.7, price="12.50")
        result = _sanitize([original])
        assert len(result) == 3
        codes = [r.fsc_code for r in result]
        assert codes == ["01088", "01090", "01091"]
        # All split records share the same description and price
        for r in result:
            assert r.fsc_description == original.fsc_description
            assert r.price == original.price

    def test_drop_section_numbering_single_digit(self):
        """Pure numeric codes 1-9 (section headers) are dropped."""
        result = _sanitize([_r("1", 0.9), _r("5", 0.8), _r("9", 0.7)])
        assert result == []

    def test_drop_section_numbering_two_digit(self):
        """Pure numeric codes 10-99 (section headers) are dropped."""
        result = _sanitize([_r("10.", 0.9), _r("12", 0.8), _r("99.", 0.7)])
        assert result == []

    def test_drop_section_numbering_with_period(self):
        """Codes like '2.' match the section-numbering pattern and are dropped."""
        result = _sanitize([_r("2.", 0.9)])
        assert result == []

    def test_drop_empty_after_strip(self):
        """Records that become empty after stripping are dropped."""
        result = _sanitize([_r("# ", 0.9), _r("*", 0.8), _r("  ", 0.7)])
        assert result == []

    def test_valid_codes_pass_through_unchanged(self):
        """Well-formed codes are returned without modification."""
        codes_in = ["A001", "01234", "E190", "B0010"]
        result = _sanitize([_r(c, 0.9) for c in codes_in])
        assert [r.fsc_code for r in result] == codes_in

    def test_sanitize_then_dedup_collapses_duplicates(self):
        """'# E190' and 'E190' normalise to the same code; dedup keeps one."""
        records = [_r("# E190", 0.7), _r("E190", 0.95)]
        sanitized = _sanitize(records)
        deduped = _dedup(sanitized)
        assert len(deduped) == 1
        assert deduped[0].fsc_code == "E190"
        assert deduped[0].extraction_confidence == 0.95

    @pytest.mark.parametrize("code", ["A001", "G003", "00001", "E003A"])
    def test_three_digit_plus_codes_not_dropped(self, code: str):
        """Codes with 3+ digits are never treated as section numbers."""
        result = _sanitize([_r(code, 0.9)])
        assert len(result) == 1
        assert result[0].fsc_code == code

    # -----------------------------------------------------------------------
    # New garbage-rejection rules (4 patterns)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("code", ["A", "a", "b", "Z"])
    def test_drop_single_letter_codes(self, code: str):
        """Single alphabetic characters are dropped (section headers, not fee codes)."""
        result = _sanitize([_r(code, 0.9)])
        assert result == [], f"Expected {code!r} to be dropped"

    @pytest.mark.parametrize("code", ["C.10.", "A.1.", "D.9.2.4", "B.3.1.", "Z.12"])
    def test_drop_dotted_section_numbers(self, code: str):
        """Letter + dot + numeric section patterns are dropped."""
        result = _sanitize([_r(code, 0.9)])
        assert result == [], f"Expected {code!r} to be dropped"

    @pytest.mark.parametrize("code", ["32.1", "32.2", "5.10", "100.25"])
    def test_drop_dotted_decimal_numbers(self, code: str):
        """Dotted decimal numbers like 32.1 are dropped (not fee codes)."""
        result = _sanitize([_r(code, 0.9)])
        assert result == [], f"Expected {code!r} to be dropped"

    @pytest.mark.parametrize("code", ["MOH", "MRI", "CPSO", "OHIP", "AMA", "BCMA"])
    def test_drop_all_caps_acronyms(self, code: str):
        """3-4 letter all-caps acronyms (no digits) are dropped."""
        result = _sanitize([_r(code, 0.9)])
        assert result == [], f"Expected {code!r} to be dropped"

    # -----------------------------------------------------------------------
    # Keep cases: verify legit codes are NOT dropped by the new rules
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("code", ["GP", "IC"])
    def test_keep_two_letter_caps_codes(self, code: str):
        """2-letter all-caps codes are kept (ambiguous; downstream handles them)."""
        result = _sanitize([_r(code, 0.9)])
        assert len(result) == 1
        assert result[0].fsc_code == code

    @pytest.mark.parametrize("code", ["A001A", "E190A"])
    def test_keep_codes_with_trailing_letter(self, code: str):
        """Codes with a trailing letter suffix (ON variant) are kept."""
        result = _sanitize([_r(code, 0.9)])
        assert len(result) == 1
        assert result[0].fsc_code == code

    @pytest.mark.parametrize("code", ["B00010", "CV07404"])
    def test_keep_bc_style_codes(self, code: str):
        """BC-style codes (leading letter + 5-digit numeric) are kept."""
        result = _sanitize([_r(code, 0.9)])
        assert len(result) == 1
        assert result[0].fsc_code == code

    @pytest.mark.parametrize("code", ["123456", "001234", "99999"])
    def test_keep_numeric_codes_up_to_6_digits(self, code: str):
        """Numeric codes up to 6 digits (legit YT) are kept."""
        result = _sanitize([_r(code, 0.9)])
        assert len(result) == 1
        assert result[0].fsc_code == code
