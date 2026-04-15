from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.yukon import YT_CODE_REGEX, YukonExtractor


def _blk(page, text, size=10.0, y=100.0, x=72.0):
    return PageBlock(
        page=page,
        text=text,
        x0=x,
        y0=y,
        x1=x + 50,
        y1=y + 10,
        font="Helvetica",
        size=size,
    )


def test_code_regex_yt_four_digits():
    assert YT_CODE_REGEX.fullmatch("0615")
    assert YT_CODE_REGEX.fullmatch("9999")
    assert not YT_CODE_REGEX.fullmatch("01712")
    assert not YT_CODE_REGEX.fullmatch("K040")


def test_extract_yt_row():
    pages = [
        _blk(1, "GENERAL PRACTICE", size=13, y=40),
        _blk(1, "0615", size=9, y=100, x=72),
        _blk(1, "Office visit", size=9, y=100, x=140),
        _blk(1, "55.00", size=9, y=100, x=500),
    ]
    rows = list(YukonExtractor().extract(pages, source_pdf_hash="c" * 64))
    assert len(rows) == 1
    assert rows[0].province == "YT"
    assert rows[0].fsc_code == "0615"
    assert rows[0].price == Decimal("55.00")


def test_extract_yt_integer_price():
    """YT has integer prices in some sections; must be accepted via Task 6 fix."""
    pages = [
        _blk(1, "GENERAL PRACTICE", size=13, y=40),
        _blk(1, "0615", size=9, y=100, x=72),
        _blk(1, "Flat call fee", size=9, y=100, x=140),
        _blk(1, "110", size=9, y=100, x=500),
    ]
    rows = list(YukonExtractor().extract(pages, source_pdf_hash="c" * 64))
    assert len(rows) == 1
    assert rows[0].price == Decimal("110")
