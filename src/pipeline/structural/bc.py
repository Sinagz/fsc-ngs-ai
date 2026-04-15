"""BC MSC Payment Schedule — structural extractor.
Verified against the 2024-2025 edition.

Subclasses OntarioExtractor for its column-grid helpers (_try_row,
_group_into_rows, _is_chapter, etc.) but overrides ``extract`` to use
the legacy column-grid path, since BC's PDF keeps the code in its own
span at x=72 — the pattern OntarioExtractor's main ``extract`` no
longer targets (ON 2025 fuses code + description into one span).

The only BC-specific overrides are the 5-digit code regex, the chapter
font size (13.0 vs 14.0), and the PROVINCE label.
"""
from __future__ import annotations

import re
from typing import ClassVar, Iterator

from src.pipeline.schema import CandidateRow, PageBlock, Province
from src.pipeline.structural.ontario import OntarioExtractor

BC_CODE_REGEX = re.compile(r"\d{5}")


class BCExtractor(OntarioExtractor):
    PROVINCE: ClassVar[Province] = "BC"
    CODE_REGEX: ClassVar[re.Pattern[str]] = BC_CODE_REGEX
    CHAPTER_FONT_SIZE: ClassVar[float] = 13.0

    def extract(
        self, pages: list[PageBlock], source_pdf_hash: str
    ) -> Iterator[CandidateRow]:
        return self._extract_column_grid(pages, source_pdf_hash)
