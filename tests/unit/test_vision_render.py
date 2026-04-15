from pathlib import Path

from src.pipeline.vision.render import (
    COLORSPACE,
    IMAGE_FORMAT,
    PAGE_DPI,
    render_page,
)

FIXTURE = Path("tests/fixtures/mini_pdf.pdf")


def test_render_returns_png_bytes():
    img = render_page(FIXTURE, page_index=0)
    assert img[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_render_is_deterministic():
    """Cache-key invariant: same PDF + page -> byte-identical bytes."""
    a = render_page(FIXTURE, page_index=1)
    b = render_page(FIXTURE, page_index=1)
    assert a == b


def test_pinned_constants():
    assert PAGE_DPI == 144
    assert IMAGE_FORMAT == "png"
    # csRGB constant exists as pymupdf.csRGB
    import pymupdf

    assert COLORSPACE is pymupdf.csRGB
