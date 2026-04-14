"""LLM rescue path for low-confidence structural rows.
Only rows with confidence < threshold (default 0.8) are sent to the LLM."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Protocol

from pydantic import BaseModel, Field

from src.pipeline.schema import CandidateRow


class RescueOutput(BaseModel):
    fsc_fn: str = ""
    fsc_description: str = ""
    price: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    resolved: bool


class _ClientLike(Protocol):
    def chat_json(self, *, prompt: str, schema: type, model: str,
                  system: str | None = None, temperature: float = 0.0) -> BaseModel: ...


SYSTEM_PROMPT = (
    "You extract Canadian physician fee codes from PDF text. "
    "Return strictly structured JSON matching the schema. "
    "If you cannot confidently extract the fields, set resolved=false."
)

USER_TEMPLATE = """Province: {province}
Code: {code}
Page: {page}

Surrounding text:
---
{context}
---

Extract fsc_fn (short name), fsc_description (full text), and price (decimal number)."""


def rescue(
    rows: list[CandidateRow],
    *,
    client: _ClientLike,
    model: str,
    context_lines: dict[tuple[int, str], str],
    threshold: float = 0.8,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    """Return (rescued_rows, unresolved_rows).
    Rows with confidence >= threshold pass through unchanged."""
    rescued: list[CandidateRow] = []
    unresolved: list[CandidateRow] = []
    for row in rows:
        if row.confidence >= threshold:
            rescued.append(row)
            continue

        context = context_lines.get((row.page, row.fsc_code), "")
        prompt = USER_TEMPLATE.format(
            province=row.province, code=row.fsc_code, page=row.page, context=context
        )
        out = client.chat_json(
            prompt=prompt, schema=RescueOutput, model=model,
            system=SYSTEM_PROMPT, temperature=0.0,
        )
        if not out.resolved:
            unresolved.append(row)
            continue

        price: Decimal | None = None
        if out.price:
            try:
                price = Decimal(out.price.replace("$", "").replace(",", ""))
            except InvalidOperation:
                price = None

        rescued.append(row.model_copy(update={
            "fsc_fn": out.fsc_fn,
            "fsc_description": out.fsc_description,
            "price": price,
            "confidence": max(row.confidence, out.confidence),
        }))
    return rescued, unresolved
