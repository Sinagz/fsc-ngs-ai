"""Strict gate between CandidateRow and FeeCodeRecord.
Emits per-reject reasons for diagnostics."""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError as PydanticValidationError

from src.pipeline.schema import CandidateRow, FeeCodeRecord, Province

PROVINCE_REGEX: dict[Province, re.Pattern[str]] = {
    "ON": re.compile(r"[A-Z]\d{3,4}"),
    "BC": re.compile(r"\d{5}"),
    "YT": re.compile(r"\d{4}"),
}


@dataclass(frozen=True)
class ValidationReject:
    candidate: CandidateRow
    reason: str


def validate(
    rows: list[CandidateRow],
) -> tuple[list[FeeCodeRecord], list[ValidationReject]]:
    seen: dict[tuple[Province, str], CandidateRow] = {}
    records: list[FeeCodeRecord] = []
    rejects: list[ValidationReject] = []

    for row in rows:
        pattern = PROVINCE_REGEX[row.province]
        if not pattern.fullmatch(row.fsc_code):
            rejects.append(ValidationReject(row, f"code regex mismatch for {row.province}"))
            continue

        key = (row.province, row.fsc_code)
        if key in seen:
            rejects.append(ValidationReject(
                row, f"duplicate {row.province}:{row.fsc_code} (first seen page {seen[key].page})"
            ))
            continue

        if row.price is not None and row.price <= 0:
            rejects.append(ValidationReject(row, "price must be positive"))
            continue

        try:
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
                extraction_method=row.origin,
                extraction_confidence=row.confidence,
            )
        except PydanticValidationError as exc:
            rejects.append(ValidationReject(row, f"schema construction failed: {exc}"))
            continue

        seen[key] = row
        records.append(record)

    return records, rejects
