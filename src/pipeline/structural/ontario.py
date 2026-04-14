"""Ontario MOH Schedule of Benefits — structural extractor.

Rules verified against the 2024-12 edition. If the publisher ships a new
edition, re-check CHAPTER_FONT_SIZE, BODY_FONT_SIZE, and the column x-ranges
before trusting the output.

Subclassing note (Tasks 7/8):
    Font sizes, column x-ranges, and tolerances are declared as ``ClassVar``s
    on ``OntarioExtractor`` so BC/YT subclasses can override them via class
    attributes. All methods read these through ``self.X`` — do not reference
    the module-level constants from inside methods.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import ClassVar, Iterator

from src.pipeline.schema import CandidateRow, PageBlock, Province
from src.pipeline.structural.base import (
    Confidence, StructuralExtractor, group_by_page,
)

ON_CODE_REGEX = re.compile(r"[A-Z]\d{3,4}")

# Module-level constants double as the documented ON defaults. Methods read
# from ``self.*`` equivalents (ClassVars) so subclasses can override cleanly.
CHAPTER_FONT_SIZE = 14.0
SECTION_FONT_SIZE = 11.5
BODY_FONT_SIZE = 10.0
FONT_TOLERANCE = 0.5

CODE_COLUMN_X = 72.0
CODE_COLUMN_TOLERANCE = 6.0

# y-tolerance: spans on the "same row" share y0 within this many points
ROW_Y_TOLERANCE = 2.5


class OntarioExtractor:
    PROVINCE: ClassVar[Province] = "ON"
    CODE_REGEX: ClassVar[re.Pattern[str]] = ON_CODE_REGEX
    CHAPTER_FONT_SIZE: ClassVar[float] = CHAPTER_FONT_SIZE
    SECTION_FONT_SIZE: ClassVar[float] = SECTION_FONT_SIZE
    BODY_FONT_SIZE: ClassVar[float] = BODY_FONT_SIZE
    FONT_TOLERANCE: ClassVar[float] = FONT_TOLERANCE
    CODE_COLUMN_X: ClassVar[float] = CODE_COLUMN_X
    CODE_COLUMN_TOLERANCE: ClassVar[float] = CODE_COLUMN_TOLERANCE
    ROW_Y_TOLERANCE: ClassVar[float] = ROW_Y_TOLERANCE

    def extract(
        self, pages: list[PageBlock], source_pdf_hash: str
    ) -> Iterator[CandidateRow]:
        """Single-pass ordered walk per page.

        Chapter/section state updates are interleaved with body-row accumulation
        so rows are tagged with the chapter/section active at the moment the
        row closes (either via a header transition or a new y-line). Rows are
        flushed before the state change, so the flushed row still sees the
        prior chapter/section via closure-by-reference on the enclosing scope.
        """
        grouped = group_by_page(pages)
        chapter: str | None = None
        section: str | None = None

        for page_num, blocks in grouped.items():
            pending_row: list[PageBlock] = []
            pending_y: float | None = None

            def _flush_row() -> CandidateRow | None:
                if not pending_row:
                    return None
                return self._try_row(
                    pending_row, page_num, chapter, section, source_pdf_hash
                )

            for block in blocks:
                if self._is_chapter(block):
                    cr = _flush_row()
                    if cr is not None:
                        yield cr
                    pending_row = []
                    pending_y = None
                    chapter = block.text.strip()
                    section = None
                    continue
                if self._is_section(block):
                    cr = _flush_row()
                    if cr is not None:
                        yield cr
                    pending_row = []
                    pending_y = None
                    section = block.text.strip()
                    continue
                if abs(block.size - self.BODY_FONT_SIZE) > self.FONT_TOLERANCE:
                    continue
                if pending_y is None or abs(block.y0 - pending_y) <= self.ROW_Y_TOLERANCE:
                    pending_row.append(block)
                    if pending_y is None:
                        pending_y = block.y0
                else:
                    cr = _flush_row()
                    if cr is not None:
                        yield cr
                    pending_row = [block]
                    pending_y = block.y0

            cr = _flush_row()
            if cr is not None:
                yield cr

    def _is_chapter(self, b: PageBlock) -> bool:
        return abs(b.size - self.CHAPTER_FONT_SIZE) <= self.FONT_TOLERANCE and b.text.isupper()

    def _is_section(self, b: PageBlock) -> bool:
        return abs(b.size - self.SECTION_FONT_SIZE) <= self.FONT_TOLERANCE

    def _group_into_rows(self, blocks: list[PageBlock]) -> list[list[PageBlock]]:
        """Legacy helper kept for callers that want to group body blocks outside
        of ``extract``. Not used by ``extract`` itself, which does interleaved
        row accumulation for correct header/row ordering."""
        rows: list[list[PageBlock]] = []
        current: list[PageBlock] = []
        current_y: float | None = None
        for b in blocks:
            if abs(b.size - self.BODY_FONT_SIZE) > self.FONT_TOLERANCE:
                continue
            if current_y is None or abs(b.y0 - current_y) <= self.ROW_Y_TOLERANCE:
                current.append(b)
                current_y = b.y0 if current_y is None else current_y
            else:
                rows.append(current)
                current = [b]
                current_y = b.y0
        if current:
            rows.append(current)
        return rows

    def _try_row(
        self,
        blocks: list[PageBlock],
        page_num: int,
        chapter: str | None,
        section: str | None,
        source_pdf_hash: str,
    ) -> CandidateRow | None:
        code_block = next(
            (b for b in blocks
             if self.CODE_REGEX.fullmatch(b.text)
             and abs(b.x0 - self.CODE_COLUMN_X) <= self.CODE_COLUMN_TOLERANCE),
            None,
        )
        if code_block is None:
            return None

        other = [b for b in blocks if b is not code_block]
        description_blocks = [
            b for b in other
            if b.x0 > self.CODE_COLUMN_X + self.CODE_COLUMN_TOLERANCE
        ]
        description_blocks.sort(key=lambda b: b.x0)

        price: Decimal | None = None
        desc_parts: list[str] = []
        for b in description_blocks:
            if self._looks_like_price(b.text):
                try:
                    price = Decimal(b.text.replace(",", "").replace("$", ""))
                    continue
                except InvalidOperation:
                    pass
            desc_parts.append(b.text)

        description = " ".join(desc_parts).strip()
        fsc_fn = description.split(".")[0][:80] if description else ""

        if description and price is not None:
            confidence = Confidence.HIGH
        elif description or price is not None:
            confidence = Confidence.MISSING_FIELD
        else:
            confidence = Confidence.AMBIGUOUS

        return CandidateRow(
            province=self.PROVINCE,
            fsc_code=code_block.text,
            fsc_fn=fsc_fn,
            fsc_description=description,
            fsc_chapter=chapter,
            fsc_section=section,
            price=price,
            page=page_num,
            source_pdf_hash=source_pdf_hash,
            confidence=confidence,
        )

    @staticmethod
    def _looks_like_price(text: str) -> bool:
        cleaned = text.replace(",", "").replace("$", "").strip()
        if not cleaned:
            return False
        try:
            value = Decimal(cleaned)
        except InvalidOperation:
            return False
        return value > 0
