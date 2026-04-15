"""Tests for the 2025 Ontario extractor.

The ON 2025-03-19 PDF edition fuses the fsc_code and description into a
single Arial 12pt span at x0 ~= 43.3, with prices in a separate span on
the same y-line at x0 > 460. Chapters are Arial,Bold 18pt uppercase;
sections are Arial,Bold 12pt. Tests mirror that layout.
"""
from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.ontario import ON_CODE_REGEX, OntarioExtractor


def _body(page, text, *, y=100.0, x=43.3):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x + 300, y1=y + 10,
                     font="Arial", size=12.0)


def _bold(page, text, *, size=12.0, y=60.0, x=43.3):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x + 200, y1=y + 10,
                     font="Arial,Bold", size=size)


def _price(page, text, *, y=100.0, x=480.0):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x + 40, y1=y + 10,
                     font="Arial", size=12.0)


def test_code_regex_matches_on_patterns():
    for code in ["K040", "A007", "Z432", "K9999"]:
        assert ON_CODE_REGEX.fullmatch(code)
    for bad in ["01712", "0615", "KK40", "1234"]:
        assert not ON_CODE_REGEX.fullmatch(bad)


def test_extract_simple_row_high_confidence():
    pages = [
        _bold(1, "GENERAL PREAMBLE", size=18, y=40),
        _body(1, "K040 Periodic health visit", y=100),
        _price(1, "78.45", y=100),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    row = rows[0]
    assert row.fsc_code == "K040"
    assert "Periodic" in row.fsc_description
    assert row.price == Decimal("78.45")
    assert row.fsc_chapter == "GENERAL PREAMBLE"
    assert row.confidence >= 0.8
    assert row.province == "ON"


def test_extract_missing_price_lower_confidence():
    pages = [
        _bold(1, "CHAPTER", size=18, y=40),
        _body(1, "K040 Periodic visit", y=100),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    assert rows[0].confidence < 0.8
    assert rows[0].price is None


def test_extract_skips_non_code_rows():
    pages = [
        _bold(1, "CHAPTER", size=18, y=40),
        _body(1, "See notes."),
    ]
    assert list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64)) == []


def test_dotted_leader_stripped_from_description():
    pages = [
        _bold(1, "CHAPTER", size=18, y=40),
        _body(1, "K040 Consultation....................................", y=100),
        _price(1, "164.90", y=100),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    assert "...." not in rows[0].fsc_description
    assert rows[0].fsc_description.strip() == "Consultation"
    assert rows[0].price == Decimal("164.90")


def test_mid_page_section_transition_tags_rows_correctly():
    """Rows before a section header get the prior section, rows after get the new."""
    pages = [
        _bold(1, "CHAPTER", size=18, y=40),
        _bold(1, "SECTION A", size=12, y=60),
        _body(1, "K040 First code", y=100),
        _price(1, "10.00", y=100),
        _bold(1, "SECTION B", size=12, y=130),
        _body(1, "K041 Second code", y=160),
        _price(1, "20.00", y=160),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 2
    assert rows[0].fsc_code == "K040"
    assert rows[0].fsc_section == "SECTION A"
    assert rows[1].fsc_code == "K041"
    assert rows[1].fsc_section == "SECTION B"


def test_integer_price_accepted():
    pages = [
        _bold(1, "CHAPTER", size=18, y=40),
        _body(1, "K040 Flat fee visit", y=100),
        _price(1, "110", y=100),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    assert rows[0].price == Decimal("110")


def test_code_column_tolerance_rejects_references_inside_prose():
    """A code-like string appearing mid-paragraph (not at the code column)
    must not be extracted as a row."""
    pages = [
        _bold(1, "CHAPTER", size=18, y=40),
        # x0=200 is well outside the code column (x0 ~= 43.3)
        _body(1, "K040 is eligible in combination with K041", y=100, x=200.0),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert rows == []
