from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.ontario import OntarioExtractor, ON_CODE_REGEX


def _blk(page, text, size=10.0, y=100.0, x=72.0):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x+50, y1=y+10,
                     font="Helvetica", size=size)


def test_code_regex_matches_on_patterns():
    for code in ["K040", "A007", "Z432", "K9999"]:
        assert ON_CODE_REGEX.fullmatch(code)
    for bad in ["01712", "0615", "KK40", "1234"]:
        assert not ON_CODE_REGEX.fullmatch(bad)


def test_extract_simple_row_high_confidence():
    pages = [
        _blk(1, "GENERAL PREAMBLE", size=14, y=40),
        _blk(1, "K040", size=10, y=100, x=72),
        _blk(1, "Periodic health visit", size=10, y=100, x=140),
        _blk(1, "78.45", size=10, y=100, x=500),
    ]
    ex = OntarioExtractor()
    rows = list(ex.extract(pages, source_pdf_hash="a" * 64))
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
        _blk(1, "CHAPTER", size=14, y=40),
        _blk(1, "K040", size=10, y=100),
        _blk(1, "Periodic visit", size=10, y=100, x=140),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    assert rows[0].confidence < 0.8
    assert rows[0].price is None


def test_extract_skips_non_code_rows():
    pages = [
        _blk(1, "CHAPTER", size=14, y=40),
        _blk(1, "See notes.", size=10, y=100),
    ]
    assert list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64)) == []


def test_mid_page_section_transition_tags_rows_correctly():
    """Regression test for the two-pass bug: rows before a section header
    should have the prior section, rows after should have the new section."""
    pages = [
        _blk(1, "CHAPTER", size=14, y=40),
        _blk(1, "SECTION A", size=11.5, y=60),
        _blk(1, "K040", size=10, y=100, x=72),
        _blk(1, "First code", size=10, y=100, x=140),
        _blk(1, "10.00", size=10, y=100, x=500),
        _blk(1, "SECTION B", size=11.5, y=130),
        _blk(1, "K041", size=10, y=160, x=72),
        _blk(1, "Second code", size=10, y=160, x=140),
        _blk(1, "20.00", size=10, y=160, x=500),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 2
    # Row order: K040 first (lower y=100), K041 second (y=160)
    assert rows[0].fsc_code == "K040"
    assert rows[0].fsc_section == "SECTION A"
    assert rows[1].fsc_code == "K041"
    assert rows[1].fsc_section == "SECTION B"


def test_integer_price_accepted():
    """YT has integer-only prices; _looks_like_price must accept them."""
    pages = [
        _blk(1, "CHAPTER", size=14, y=40),
        _blk(1, "K040", size=10, y=100, x=72),
        _blk(1, "Flat fee visit", size=10, y=100, x=140),
        _blk(1, "110", size=10, y=100, x=500),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    from decimal import Decimal
    assert rows[0].price == Decimal("110")
