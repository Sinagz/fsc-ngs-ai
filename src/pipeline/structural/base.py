"""Shared utilities and Protocol for per-province structural extractors."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import ClassVar, Iterator, Protocol

from src.pipeline.schema import CandidateRow, PageBlock, Province


class Confidence:
    """Explicit confidence bands. Spec: rescue threshold is 0.8."""
    HIGH = 1.0               # all fields matched cleanly
    ADJACENT_FIELD = 0.85    # one field recovered from adjacent line
    MISSING_FIELD = 0.6      # matched code regex, missing price or description
    AMBIGUOUS = 0.3          # possibly a table header
    REJECT = 0.0


class StructuralExtractor(Protocol):
    PROVINCE: ClassVar[Province]
    CODE_REGEX: ClassVar[re.Pattern[str]]

    def extract(
        self, pages: list[PageBlock], source_pdf_hash: str
    ) -> Iterator[CandidateRow]: ...


def group_by_page(blocks: list[PageBlock]) -> dict[int, list[PageBlock]]:
    grouped: dict[int, list[PageBlock]] = defaultdict(list)
    for b in blocks:
        grouped[b.page].append(b)
    for page in grouped:
        grouped[page].sort(key=lambda b: (b.y0, b.x0))
    return dict(sorted(grouped.items()))


def is_section_header(block: PageBlock, *, body_size: float, tolerance: float = 0.5) -> bool:
    """Row-level check: True if this block's font is large enough to look like a header
    relative to known body text size. Use while walking a page when body_size is known.
    Effective cutoff at default tolerance: block.size >= body_size + 1.0."""
    return block.size >= body_size + 1.5 - tolerance and len(block.text) >= 3


def detect_section_headers(
    blocks: list[PageBlock], *, header_size: float, tolerance: float = 0.3
) -> list[tuple[int, str]]:
    """Bulk scan: return (page, text) for every block whose font size matches
    `header_size` exactly (within `tolerance`). Use when the header font size is
    known a priori from the PDF's styling conventions."""
    return [
        (b.page, b.text)
        for b in blocks
        if abs(b.size - header_size) <= tolerance
    ]
