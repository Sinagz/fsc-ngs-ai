from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.bc import BCExtractor, BC_CODE_REGEX


def _blk(page, text, size=10.0, y=100.0, x=72.0):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x+50, y1=y+10,
                     font="Helvetica", size=size)


def test_code_regex_bc_five_digits():
    assert BC_CODE_REGEX.fullmatch("01712")
    assert BC_CODE_REGEX.fullmatch("99999")
    assert not BC_CODE_REGEX.fullmatch("K040")
    assert not BC_CODE_REGEX.fullmatch("9999")
    assert not BC_CODE_REGEX.fullmatch("012345")


def test_extract_bc_row():
    pages = [
        _blk(1, "CONSULTATIONS", size=13, y=40),
        _blk(1, "01712", size=10, y=100, x=72),
        _blk(1, "Limited consultation", size=10, y=100, x=140),
        _blk(1, "45.20", size=10, y=100, x=500),
    ]
    rows = list(BCExtractor().extract(pages, source_pdf_hash="b" * 64))
    assert len(rows) == 1
    r = rows[0]
    assert r.province == "BC"
    assert r.fsc_code == "01712"
    assert r.price == Decimal("45.20")
    assert r.confidence >= 0.8
    assert r.fsc_chapter == "CONSULTATIONS"


def test_bc_inherits_mid_page_section_fix():
    """BC must inherit the single-pass row emission from Ontario so sections
    are tagged correctly. Regression guard against BC reintroducing the bug."""
    pages = [
        _blk(1, "CONSULTATIONS", size=13, y=40),
        _blk(1, "SECTION A", size=11.5, y=60),
        _blk(1, "01712", size=10, y=100, x=72),
        _blk(1, "First", size=10, y=100, x=140),
        _blk(1, "10.00", size=10, y=100, x=500),
        _blk(1, "SECTION B", size=11.5, y=130),
        _blk(1, "02000", size=10, y=160, x=72),
        _blk(1, "Second", size=10, y=160, x=140),
        _blk(1, "20.00", size=10, y=160, x=500),
    ]
    rows = list(BCExtractor().extract(pages, source_pdf_hash="b" * 64))
    assert len(rows) == 2
    assert rows[0].fsc_code == "01712"
    assert rows[0].fsc_section == "SECTION A"
    assert rows[1].fsc_code == "02000"
    assert rows[1].fsc_section == "SECTION B"
