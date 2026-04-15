"""Section-context map from pymupdf TOC. Full implementation in Task 6."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionContext:
    chapter: str | None
    section: str | None
    subsection: str | None
