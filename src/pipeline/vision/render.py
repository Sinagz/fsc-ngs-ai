"""PDF page rendering with pinned, cache-keyed constants.

DO NOT EDIT PAGE_DPI, COLORSPACE, or IMAGE_FORMAT without bumping
``schema_version``. These three values are part of the hishel cache key
(via the request body bytes), so any change silently invalidates every
cached vision call.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_DPI = 144
COLORSPACE = pymupdf.csRGB
IMAGE_FORMAT = "png"


def render_page(pdf_path: Path, *, page_index: int) -> bytes:
    """Render a single page as PNG bytes. ``page_index`` is 0-based."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=PAGE_DPI, colorspace=COLORSPACE)
        return pix.tobytes(IMAGE_FORMAT)
    finally:
        doc.close()
