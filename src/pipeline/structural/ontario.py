"""Ontario MOH Schedule of Benefits — structural extractor.

Rules verified against the 2025-03-19 edition (Nov 18, 2025 reprint).

Layout summary (2025 edition):
    * Body codes live in Arial 12pt spans at x0 approx 43.3, with the
      fsc_code and description fused in the same span text
      (e.g. ``"K887 CTO initiation including completion..."``).
    * Prices appear in a separate span on the same y-line at x0 > 460,
      typically after a dotted leader inside the description span.
    * Chapters are Arial/Times Bold 17-18pt; sections are Arial,Bold 12pt.

Legacy note:
    The prior 2024-12 edition rendered each code in its own span at
    x0 = 72, separate from description + price spans. That column-grid
    layout still applies to the BC and YT PDFs, so the helper method
    ``_extract_column_grid`` preserves it and BCExtractor overrides
    ``extract`` to call that helper.

Subclassing note:
    BC and YT subclass OntarioExtractor purely for shared config
    constants (CODE_COLUMN_TOLERANCE, ROW_Y_TOLERANCE, price parser).
    BC overrides ``extract`` to use ``_extract_column_grid``.
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
# Splits "E400B Evenings (17:00h - 24:00h)..." into code + remainder.
ON_INLINE_CODE_SPLIT = re.compile(r"^([A-Z]\d{3,4})\s+(.+)$")
_DOTTED_LEADER = re.compile(r"[.\u2024\u2025\u2026]{3,}")

# 2025 edition constants
ON_BODY_FONT_SIZE = 12.0
ON_CHAPTER_FONT_SIZE = 18.0
ON_SECTION_FONT_NAME_SUBSTR = "Bold"  # section headers use Arial,Bold 12pt
ON_CODE_COLUMN_X = 43.3
ON_CODE_COLUMN_TOLERANCE = 3.0
ON_PRICE_COLUMN_MIN_X = 460.0
ON_FONT_TOLERANCE = 0.5
ON_ROW_Y_TOLERANCE = 2.5

# Legacy column-grid defaults (used by BC/YT via _extract_column_grid)
CHAPTER_FONT_SIZE = 14.0
SECTION_FONT_SIZE = 11.5
BODY_FONT_SIZE = 10.0
FONT_TOLERANCE = 0.5
CODE_COLUMN_X = 72.0
CODE_COLUMN_TOLERANCE = 6.0
ROW_Y_TOLERANCE = 2.5


class OntarioExtractor:
    PROVINCE: ClassVar[Province] = "ON"
    CODE_REGEX: ClassVar[re.Pattern[str]] = ON_CODE_REGEX

    # Legacy column-grid ClassVars (BC/YT override)
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
        """Extract ON fee codes using the 2025 inline-span layout.

        Walks each page, tracking the current chapter/section. For each
        body-font span at the code column whose text starts with a code
        regex match, emits a CandidateRow with the remainder as
        description and, if present, a price span on the same y-line.
        """
        grouped = group_by_page(pages)
        chapter: str | None = None
        section: str | None = None

        for page_num, blocks in grouped.items():
            # Pre-scan y-lines for price lookups
            by_y: dict[float, list[PageBlock]] = {}
            for b in blocks:
                y_key = round(b.y0, 1)
                by_y.setdefault(y_key, []).append(b)

            for b in blocks:
                if self._is_on_chapter(b):
                    chapter = b.text.strip()
                    section = None
                    continue
                if self._is_on_section(b):
                    section = b.text.strip()
                    continue
                if not self._is_on_body(b):
                    continue
                if abs(b.x0 - ON_CODE_COLUMN_X) > ON_CODE_COLUMN_TOLERANCE:
                    continue

                m = ON_INLINE_CODE_SPLIT.match(b.text)
                if m is None:
                    continue

                code = m.group(1)
                rest = m.group(2)

                # Strip dotted leaders and any trailing whitespace.
                rest_clean = _DOTTED_LEADER.sub(" ", rest).strip()
                # If the price is in the same span (rare for 2025 ON),
                # try to peel it off the end.
                price: Decimal | None = None
                desc = rest_clean
                tail = rest_clean.rsplit(" ", 1)
                if len(tail) == 2 and self._looks_like_price(tail[1]):
                    price = _to_decimal(tail[1])
                    desc = tail[0].strip()

                # Otherwise look in neighbouring spans on the same y-line.
                if price is None:
                    price = self._find_price_on_line(b, by_y)

                description = desc
                fsc_fn = description.split(".")[0][:80] if description else ""

                if description and price is not None:
                    confidence = Confidence.HIGH
                elif description or price is not None:
                    confidence = Confidence.MISSING_FIELD
                else:
                    confidence = Confidence.AMBIGUOUS

                yield CandidateRow(
                    province=self.PROVINCE,
                    fsc_code=code,
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
    def _is_on_chapter(b: PageBlock) -> bool:
        return (
            abs(b.size - ON_CHAPTER_FONT_SIZE) <= ON_FONT_TOLERANCE + 1.0
            and "Bold" in b.font
            and b.text.strip().isupper()
            and len(b.text.strip()) >= 3
        )

    @staticmethod
    def _is_on_section(b: PageBlock) -> bool:
        return (
            abs(b.size - ON_BODY_FONT_SIZE) <= ON_FONT_TOLERANCE
            and ON_SECTION_FONT_NAME_SUBSTR in b.font
            and len(b.text.strip()) >= 3
            and not ON_INLINE_CODE_SPLIT.match(b.text)
        )

    @staticmethod
    def _is_on_body(b: PageBlock) -> bool:
        return (
            abs(b.size - ON_BODY_FONT_SIZE) <= ON_FONT_TOLERANCE
            and ON_SECTION_FONT_NAME_SUBSTR not in b.font
        )

    @classmethod
    def _find_price_on_line(
        cls, code_block: PageBlock, by_y: dict[float, list[PageBlock]]
    ) -> Decimal | None:
        y_key = round(code_block.y0, 1)
        candidates: list[PageBlock] = []
        for dy in (-0.1, 0.0, 0.1):
            candidates.extend(by_y.get(round(y_key + dy, 1), []))
        # Keep spans to the right of the price column cutoff
        right_spans = [
            b for b in candidates
            if b.x0 >= ON_PRICE_COLUMN_MIN_X
            and abs(b.y0 - code_block.y0) <= ON_ROW_Y_TOLERANCE
        ]
        right_spans.sort(key=lambda b: b.x0)
        for b in right_spans:
            if cls._looks_like_price(b.text):
                return _to_decimal(b.text)
        return None

    # ---- Legacy column-grid path (BC/YT) -----------------------------------

    def _extract_column_grid(
        self, pages: list[PageBlock], source_pdf_hash: str
    ) -> Iterator[CandidateRow]:
        """Single-pass ordered walk per page, column-grid layout.

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


def _to_decimal(text: str) -> Decimal | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
