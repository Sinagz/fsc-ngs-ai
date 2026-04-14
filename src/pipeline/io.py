"""PDF loading via pymupdf, returning structured PageBlock records."""
from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # pymupdf

from src.pipeline.schema import PageBlock


def pdf_hash(path: Path) -> str:
    """SHA-256 of the raw PDF bytes. Stable across runs, changes when PDF changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pdf(path: Path) -> list[PageBlock]:
    """Load all text blocks from a PDF with layout and font metadata."""
    blocks: list[PageBlock] = []
    doc = fitz.open(str(path))
    try:
        for page_idx, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        if not text:
                            continue
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        blocks.append(PageBlock(
                            page=page_idx, text=text,
                            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                            font=span.get("font", ""), size=float(span.get("size", 0.0)),
                        ))
    finally:
        doc.close()
    return blocks
