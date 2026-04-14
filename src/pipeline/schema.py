"""Canonical pydantic schemas. Single source of truth for the pipeline."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Province = Literal["ON", "BC", "YT"]
ExtractionMethod = Literal["structural", "semantic"]
NGSMethod = Literal["exact", "llm", "manual"]


class PageBlock(BaseModel):
    """One pymupdf text block with layout metadata."""
    model_config = ConfigDict(frozen=True)

    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font: str
    size: float


class CandidateRow(BaseModel):
    """Structural-parser output before validation + NGS mapping."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    province: Province
    fsc_code: str
    fsc_fn: str
    fsc_description: str
    fsc_chapter: str | None = None
    fsc_section: str | None = None
    fsc_subsection: str | None = None
    fsc_notes: str | None = None
    price: Decimal | None = None
    page: int
    source_pdf_hash: str
    confidence: float = Field(ge=0.0, le=1.0)


class FeeCodeRecord(BaseModel):
    """Validated, NGS-mapped fee code record. The canonical type."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"
    province: Province
    fsc_code: str
    fsc_fn: str
    fsc_description: str
    fsc_chapter: str | None = None
    fsc_section: str | None = None
    fsc_subsection: str | None = None
    fsc_notes: str | None = None
    price: Decimal | None = None
    page: int
    source_pdf_hash: str
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    NGS_code: str | None = None
    NGS_label: str | None = None
    NGS_mapping_method: NGSMethod | None = None
    NGS_mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NGSRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ngs_code: str
    ngs_label: str
    ngs_description: str = ""
    code_refs: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    generated_at: str
    git_sha: str
    row_counts: dict[str, int]
    source_pdf_hashes: dict[str, str]
    models: dict[str, str]
    regression_override: str | None = None
