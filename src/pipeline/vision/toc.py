"""Build a (page -> SectionContext) map from pymupdf's TOC.

pymupdf's ``get_toc()`` returns ``[[level, title, page], ...]`` where
``level`` is 1-based and ``page`` is 1-based. Entries apply forward from
their page until superseded by a same-or-higher-level entry.

Empty dict is returned when the PDF has no outline (the Yukon case).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class SectionContext:
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None


def build_section_map(pdf_path: Path) -> dict[int, SectionContext]:
    """Return a map of 1-based page number -> SectionContext.

    Missing pages (e.g. before the first L1 entry) get a SectionContext
    with all-None fields. Empty dict if the PDF has no TOC at all.
    """
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    n_pages = doc.page_count
    doc.close()

    if not toc:
        return {}

    # Sort by page, then by level so higher-level entries apply first on the
    # same page. This keeps (L1, L2, L3) state consistent when multiple
    # levels share a page.
    toc_sorted = sorted(toc, key=lambda e: (e[2], e[0]))

    result: dict[int, SectionContext] = {}
    current: list[str | None] = [None, None, None]  # indices 0..2 => L1..L3
    toc_iter = iter(toc_sorted)
    next_entry = next(toc_iter, None)

    for page in range(1, n_pages + 1):
        while next_entry is not None and next_entry[2] <= page:
            lvl, title, _ = next_entry
            if 1 <= lvl <= 3:
                current[lvl - 1] = title
                for deeper in range(lvl, 3):
                    current[deeper] = None
            next_entry = next(toc_iter, None)
        result[page] = SectionContext(
            chapter=current[0], section=current[1], subsection=current[2]
        )

    return result
