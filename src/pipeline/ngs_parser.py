"""Parse CIHI NGS reference DOCX files into NGSRecord objects."""
from __future__ import annotations

import re
from pathlib import Path

import docx

from src.pipeline.schema import NGSRecord

NGS_HEADING_RE = re.compile(r"^(\S+)\s+(.+)$")
CODE_LIST_RE = re.compile(r"[A-Z]?\d{3,5}")


def parse_ngs_docx(path: Path) -> list[NGSRecord]:
    """Heading-1 paragraphs are NGS headers ("<code> <label>").

    Body paragraphs between headers form the description; any
    "Includes codes:" line contributes to code_refs.
    """
    doc = docx.Document(str(path))
    records: list[NGSRecord] = []
    current_code: str | None = None
    current_label = ""
    desc_parts: list[str] = []
    code_refs: list[str] = []

    def _flush() -> None:
        nonlocal current_code, current_label, desc_parts, code_refs
        if current_code is not None:
            records.append(
                NGSRecord(
                    ngs_code=current_code,
                    ngs_label=current_label,
                    ngs_description=" ".join(desc_parts).strip(),
                    code_refs=tuple(dict.fromkeys(code_refs)),
                )
            )
        current_code = None
        current_label = ""
        desc_parts = []
        code_refs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            _flush()
            m = NGS_HEADING_RE.match(text)
            if m:
                current_code = m.group(1)
                current_label = m.group(2)
            continue
        if text.lower().startswith("includes codes"):
            code_refs.extend(CODE_LIST_RE.findall(text))
        else:
            desc_parts.append(text)
    _flush()
    return records
