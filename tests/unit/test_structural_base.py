from src.pipeline.structural.base import (
    Confidence, group_by_page, detect_section_headers, is_section_header,
)
from src.pipeline.schema import PageBlock


def _blk(page, text, size=10.0, y=100.0):
    return PageBlock(page=page, text=text, x0=72, y0=y, x1=100, y1=y+10,
                     font="Helvetica", size=size)


def test_confidence_constants_cover_threshold():
    assert Confidence.HIGH == 1.0
    assert Confidence.ADJACENT_FIELD == 0.85
    assert Confidence.AMBIGUOUS == 0.3
    # Threshold for LLM rescue is 0.8 per spec
    assert Confidence.ADJACENT_FIELD >= 0.8
    assert Confidence.MISSING_FIELD < 0.8


def test_group_by_page_preserves_order():
    blocks = [
        _blk(1, "A", y=100), _blk(2, "B", y=200),
        _blk(1, "C", y=150), _blk(3, "D", y=50),
    ]
    grouped = group_by_page(blocks)
    assert list(grouped.keys()) == [1, 2, 3]
    # within a page, sorted by y then x
    assert [b.text for b in grouped[1]] == ["A", "C"]


def test_is_section_header_detects_large_font():
    assert is_section_header(_blk(1, "ANAESTHESIA", size=14.0), body_size=10.0)
    assert not is_section_header(_blk(1, "K040", size=10.0), body_size=10.0)


def test_is_section_header_boundary():
    # body=10.0, default tolerance=0.5 → effective cutoff at size >= 11.0
    assert is_section_header(_blk(1, "HDR", size=11.0), body_size=10.0)
    assert not is_section_header(_blk(1, "HDR", size=10.9), body_size=10.0)
    # Text must be at least 3 chars
    assert not is_section_header(_blk(1, "AB", size=14.0), body_size=10.0)


def test_detect_section_headers_emits_pairs():
    blocks = [
        _blk(1, "CHAPTER A", size=14, y=50),
        _blk(1, "K040 Foo 78.45", size=10, y=80),
        _blk(2, "CHAPTER B", size=14, y=50),
        _blk(2, "K041 Bar 99.00", size=10, y=80),
    ]
    headers = detect_section_headers(blocks, header_size=14.0)
    assert headers == [(1, "CHAPTER A"), (2, "CHAPTER B")]
