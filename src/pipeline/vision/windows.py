"""Sliding-window builder.

For a PDF of N pages, yields N windows (one per page). Each window owns
exactly one target page; context page is the page immediately before it
(or None for window 1). Every fee-code entry lives entirely within its
owning window, so the extractor can emit exactly once using the
"emit-where-it-completes" rule (see prompts.py).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.pipeline.vision.toc import SectionContext


@dataclass(frozen=True)
class Window:
    target_page: int  # 1-based
    context_page: int | None  # 1-based, or None for the seed window
    section_hints: SectionContext


_EMPTY_SECTION = SectionContext(chapter=None, section=None, subsection=None)


def build_windows(
    *, num_pages: int, section_map: dict[int, SectionContext]
) -> Iterator[Window]:
    for page in range(1, num_pages + 1):
        yield Window(
            target_page=page,
            context_page=page - 1 if page > 1 else None,
            section_hints=section_map.get(page, _EMPTY_SECTION),
        )
