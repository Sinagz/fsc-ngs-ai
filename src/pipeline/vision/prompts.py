"""Prompt templates for the vision extraction pass.

SYSTEM_TEMPLATE is the stable instruction block. `build_prompt` fills in
the per-window variable parts (province, target page, TOC hints).
"""
from __future__ import annotations

from src.pipeline.schema import Province
from src.pipeline.vision.toc import SectionContext


SYSTEM_TEMPLATE = """\
You are extracting fee-code entries from the {province} physician fee schedule.

One or two rendered pages are attached below. The TARGET PAGE is page {target_page}.

RULES (critical - each entry has exactly one owning window):
- Emit an entry iff its final visible line appears on the TARGET PAGE.
- Skip entries cut off at the bottom of the TARGET PAGE - the next window owns them.
- Skip entries whose end lies on the earlier (non-target) page - a prior window already owned them.

SECTION CONTEXT FOR PAGE {target_page} (authoritative - use these exact strings when set):
  chapter:    {chapter}
  section:    {section}
  subsection: {subsection}
If a field is null, read whatever header is visible on the TARGET PAGE itself.

FIELD RULES:
  fsc_code              : code exactly as printed.
  fsc_fn                : short function/name string (the brief label next to the code).
  fsc_description       : full description as one string; join wrapped lines with a space.
  fsc_notes             : rich notes paragraphs attached to this code, or null if none.
  price                 : dollar amount for this code (strip the $). null if no price shown.
  page                  : {target_page} (always the TARGET PAGE number, 1-indexed).
  extraction_confidence : your calibrated 0.0-1.0 confidence for this record.
"""


def _fmt(val: str | None) -> str:
    return val if val else "null"


def build_prompt(
    *, province: Province, target_page: int, section: SectionContext
) -> str:
    """Compose the system prompt for one window."""
    return SYSTEM_TEMPLATE.format(
        province=province,
        target_page=target_page,
        chapter=_fmt(section.chapter),
        section=_fmt(section.section),
        subsection=_fmt(section.subsection),
    )
