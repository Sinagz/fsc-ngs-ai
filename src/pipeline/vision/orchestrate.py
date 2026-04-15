"""Province-level extraction orchestrator.

Glues render + TOC + windows + extract together with bounded asyncio
concurrency, then merges, sanitizes, dedups, and promotes VisionRecord ->
FeeCodeRecord.

The pymupdf Document is shared across coroutines - pymupdf is not thread
safe, but asyncio runs everything on one OS thread (the blocking work
happens inside ``asyncio.to_thread`` for the OpenAI call only). Rendering
in the loop thread is fine: ~30-50ms per page is dominated by the 5-15s
LLM latency.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path

import pymupdf
from tqdm.asyncio import tqdm_asyncio

from src.openai_client import OpenAIClient
from src.pipeline.schema import FeeCodeRecord, Province
from src.pipeline.vision.extract import extract_window
from src.pipeline.vision.render import COLORSPACE, IMAGE_FORMAT, PAGE_DPI
from src.pipeline.vision.schema import VisionRecord
from src.pipeline.vision.toc import build_section_map
from src.pipeline.vision.windows import Window, build_windows

logger = logging.getLogger(__name__)

# Matches pure section-numbering artefacts like "1", "2.", "10.", "12"
_SECTION_NUMBER_RE = re.compile(r"^\d{1,2}\.?$")


def _sanitize(records: list[VisionRecord]) -> list[VisionRecord]:
    """Normalize ``fsc_code`` fields and expand / drop pathological entries.

    Rules applied in order:
    1. Strip leading marker characters (``#``, ``*``) and surrounding
       whitespace from ``fsc_code``.
    2. Strip all internal whitespace so ``"E 190"`` becomes ``"E190"``.
    3. Split comma-separated codes into N separate records each sharing the
       same description, price, notes, and confidence as the original.
    4. Drop records whose ``fsc_code`` matches the section-numbering pattern
       (``^[0-9]{1,2}[.]?$``) -- these are table/chapter headers, not fee codes.
    5. Drop records whose ``fsc_code`` is empty after the above steps.
    """
    out: list[VisionRecord] = []
    for r in records:
        raw = r.fsc_code.lstrip("#* ").strip()
        # Remove all internal whitespace
        raw = re.sub(r"\s+", "", raw)

        # Split on commas to handle concatenated code lists
        parts = [p.strip() for p in raw.split(",") if p.strip()]

        for code in parts:
            # Drop section-numbering artefacts
            if _SECTION_NUMBER_RE.match(code):
                continue
            # Drop anything that became empty
            if not code:
                continue
            out.append(r.model_copy(update={"fsc_code": code}))

    return out


def _render_window_images(doc: pymupdf.Document, window: Window) -> list[bytes]:
    imgs: list[bytes] = []
    if window.context_page is not None:
        pix = doc[window.context_page - 1].get_pixmap(dpi=PAGE_DPI, colorspace=COLORSPACE)
        imgs.append(pix.tobytes(IMAGE_FORMAT))
    pix = doc[window.target_page - 1].get_pixmap(dpi=PAGE_DPI, colorspace=COLORSPACE)
    imgs.append(pix.tobytes(IMAGE_FORMAT))
    return imgs


def _dedup(records: list[VisionRecord]) -> list[VisionRecord]:
    """Keep highest extraction_confidence per (province, fsc_code);
    tiebreak on lowest page."""
    by_key: dict[tuple[str, str], VisionRecord] = {}
    for r in records:
        key = (r.province, r.fsc_code)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = r
            continue
        if r.extraction_confidence > prev.extraction_confidence:
            by_key[key] = r
        elif r.extraction_confidence == prev.extraction_confidence and r.page < prev.page:
            by_key[key] = r
    return list(by_key.values())


def _to_fee_code_record(
    vr: VisionRecord, *, province: Province, source_pdf_hash: str
) -> FeeCodeRecord:
    return FeeCodeRecord(
        schema_version="2",
        province=province,
        fsc_code=vr.fsc_code,
        fsc_fn=vr.fsc_fn,
        fsc_description=vr.fsc_description,
        fsc_chapter=vr.fsc_chapter,
        fsc_section=vr.fsc_section,
        fsc_subsection=vr.fsc_subsection,
        fsc_notes=vr.fsc_notes,
        price=vr.price,
        page=vr.page,
        source_pdf_hash=source_pdf_hash,
        extraction_method="vision",
        extraction_confidence=vr.extraction_confidence,
    )


def _hash_pdf(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


async def extract_province(
    pdf_path: Path,
    *,
    province: Province,
    client: OpenAIClient,
    concurrency: int = 20,
    model: str = "gpt-5.4-mini",
    failure_log: Path | None = None,
) -> list[FeeCodeRecord]:
    section_map = build_section_map(pdf_path)
    pdf_hash = _hash_pdf(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        windows = list(build_windows(num_pages=doc.page_count, section_map=section_map))
        sem = asyncio.Semaphore(concurrency)

        async def _one(w: Window) -> list[VisionRecord]:
            async with sem:
                try:
                    imgs = _render_window_images(doc, w)
                    return await extract_window(
                        window=w,
                        province=province,
                        images=imgs,
                        client=client,
                        model=model,
                        failure_log=failure_log,
                    )
                except Exception as e:  # noqa: BLE001 — any failure becomes a zero-record window
                    logger.warning(
                        "[%s] window %d raised %s: %s (skipping, contributing 0 records)",
                        province, w.target_page, type(e).__name__, e,
                    )
                    if failure_log is not None:
                        import json as _json
                        failure_log.parent.mkdir(parents=True, exist_ok=True)
                        with failure_log.open("a", encoding="utf-8") as f:
                            f.write(_json.dumps({
                                "province": province,
                                "target_page": w.target_page,
                                "error_class": type(e).__name__,
                                "message": str(e)[:500],
                            }) + "\n")
                    return []

        logger.info("[%s] dispatching %d windows (concurrency=%d)",
                    province, len(windows), concurrency)
        batches: list[list[VisionRecord]] = await tqdm_asyncio.gather(
            *[_one(w) for w in windows],
            desc=f"{province} vision extraction",
        )
    finally:
        doc.close()

    flat = [r for batch in batches for r in batch]
    sanitized = _sanitize(flat)
    dropped = len(flat) - len(sanitized)
    if dropped:
        logger.info("[%s] sanitize dropped %d records (section-numbering / empty / expanded splits)",
                    province, dropped)
    deduped = _dedup(sanitized)
    dup_count = len(sanitized) - len(deduped)
    if dup_count:
        logger.info("[%s] dedup removed %d duplicate records", province, dup_count)
    return [_to_fee_code_record(r, province=province, source_pdf_hash=pdf_hash) for r in deduped]
