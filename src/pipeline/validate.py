"""Strict gate between CandidateRow and FeeCodeRecord.
Emits per-reject reasons for diagnostics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from src.pipeline.schema import CandidateRow, FeeCodeRecord, Province

PROVINCE_REGEX: dict[Province, Pattern[str]] = {
    "ON": re.compile(r"[A-Z]\d{3,4}"),
    "BC": re.compile(r"\d{5}"),
    "YT": re.compile(r"\d{4}"),
}


@dataclass(frozen=True)
class ValidationError:
    candidate: CandidateRow
    reason: str


def validate(
    rows: list[CandidateRow],
) -> tuple[list[FeeCodeRecord], list[ValidationError]]:
    seen: dict[tuple[Province, str], CandidateRow] = {}
    records: list[FeeCodeRecord] = []
    rejects: list[ValidationError] = []

    for row in rows:
        pattern = PROVINCE_REGEX[row.province]
        if not pattern.fullmatch(row.fsc_code):
            rejects.append(ValidationError(row, f"code regex mismatch for {row.province}"))
            continue

        key = (row.province, row.fsc_code)
        if key in seen:
            rejects.append(ValidationError(
                row, f"duplicate {row.province}:{row.fsc_code} (first seen page {seen[key].page})"
            ))
            continue
        seen[key] = row

        if row.price is not None and row.price <= 0:
            rejects.append(ValidationError(row, "price must be positive"))
            continue

        record = FeeCodeRecord(
            province=row.province,
            fsc_code=row.fsc_code,
            fsc_fn=row.fsc_fn,
            fsc_description=row.fsc_description,
            fsc_chapter=row.fsc_chapter,
            fsc_section=row.fsc_section,
            fsc_subsection=row.fsc_subsection,
            fsc_notes=row.fsc_notes,
            price=row.price,
            page=row.page,
            source_pdf_hash=row.source_pdf_hash,
            extraction_method="structural" if row.confidence >= 0.8 else "semantic",
            extraction_confidence=row.confidence,
        )
        records.append(record)

    return records, rejects
