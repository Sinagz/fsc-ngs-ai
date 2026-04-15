"""Response schemas for the vision extraction pass."""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from src.pipeline.schema import Province


class VisionRecord(BaseModel):
    """One fee-code entry as emitted by the vision extractor.

    The orchestrator promotes these to :class:`FeeCodeRecord` by attaching
    ``schema_version``, ``source_pdf_hash``, and
    ``extraction_method="vision"`` (NGS fields stay null for the mapper).
    """

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
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class WindowExtraction(BaseModel):
    """Response envelope for one window-level ``chat_vision_json`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: list[VisionRecord]
