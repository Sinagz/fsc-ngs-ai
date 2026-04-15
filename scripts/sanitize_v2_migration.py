"""Migrate v2026-04-15-vision artifact by applying fsc_code sanitization.

Reads the existing artifact, applies the new _sanitize + _dedup rules to
expand/drop/normalize fsc_codes, re-maps NGS only for records whose codes
changed or were previously unmapped, regenerates embeddings, and writes the
cleaned artifact to data/parsed/v2026-04-15-vision-clean/.

Usage:
    python -m scripts.sanitize_v2_migration

Estimated cost: ~$0.08-0.12 (partial NGS remapping + embeddings).
Expected wall-clock: 5-15 min (most time in NGS LLM verdicts).
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Configure logging before importing pipeline modules so their loggers pick it up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

from src.openai_client import OpenAIClient  # noqa: E402 -- after load_dotenv
from src.pipeline.embed import build_embeddings, save_npz  # noqa: E402
from src.pipeline.ngs_mapper import map_ngs  # noqa: E402
from src.pipeline.ngs_parser import parse_ngs_docx  # noqa: E402
from src.pipeline.schema import FeeCodeRecord, Manifest  # noqa: E402
from src.pipeline.vision.orchestrate import _dedup, _sanitize  # noqa: E402
from src.pipeline.vision.schema import VisionRecord  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SRC_VERSION = "v2026-04-15-vision"
DST_VERSION = "v2026-04-15-vision-clean"

SRC_DIR = ROOT / "data" / "parsed" / SRC_VERSION
DST_DIR = ROOT / "data" / "parsed" / DST_VERSION
DOCX_DIR = ROOT / "data" / "raw" / "docx"

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
LLM_MODEL = "gpt-5.4-mini"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _fee_to_vision(r: FeeCodeRecord) -> VisionRecord:
    """Strip pipeline-only fields to recover a VisionRecord for re-sanitization."""
    return VisionRecord(
        province=r.province,
        fsc_code=r.fsc_code,
        fsc_fn=r.fsc_fn,
        fsc_description=r.fsc_description,
        fsc_chapter=r.fsc_chapter,
        fsc_section=r.fsc_section,
        fsc_subsection=r.fsc_subsection,
        fsc_notes=r.fsc_notes,
        price=r.price,
        page=r.page,
        extraction_confidence=r.extraction_confidence,
    )


def _promote(vr: VisionRecord, *, source_pdf_hash: str) -> FeeCodeRecord:
    """Promote a VisionRecord back to a FeeCodeRecord (NGS fields cleared)."""
    return FeeCodeRecord(
        schema_version="2",
        province=vr.province,
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
        # NGS fields cleared -- will be re-populated by map_ngs
        NGS_code=None,
        NGS_label=None,
        NGS_mapping_method=None,
        NGS_mapping_confidence=0.0,
    )


def _count_ngs(records: list[FeeCodeRecord]) -> int:
    return sum(1 for r in records if r.NGS_code is not None)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load source artifact
    # ------------------------------------------------------------------
    logger.info("Loading source artifact from %s", SRC_DIR)
    old_codes = json.loads((SRC_DIR / "codes.json").read_text(encoding="utf-8"))
    old_records: list[FeeCodeRecord] = [FeeCodeRecord(**d) for d in old_codes]

    old_manifest_raw = json.loads((SRC_DIR / "manifest.json").read_text(encoding="utf-8"))
    old_source_pdf_hashes: dict[str, str] = old_manifest_raw.get("source_pdf_hashes", {})

    logger.info(
        "Loaded %d records (%s)",
        len(old_records),
        {p: sum(1 for r in old_records if r.province == p) for p in ("ON", "BC", "YT")},
    )
    logger.info(
        "Old NGS mapping rate: %d / %d (%.1f%%)",
        _count_ngs(old_records),
        len(old_records),
        _count_ngs(old_records) / len(old_records) * 100 if old_records else 0,
    )

    # Build an index of old records by (province, fsc_code) for NGS carryover
    old_index: dict[tuple[str, str], FeeCodeRecord] = {
        (r.province, r.fsc_code): r for r in old_records
    }

    # ------------------------------------------------------------------
    # 2. Convert -> sanitize -> dedup (province by province to track hashes)
    # ------------------------------------------------------------------
    logger.info("Converting FeeCodeRecords back to VisionRecords for re-sanitization...")
    all_vision: list[VisionRecord] = [_fee_to_vision(r) for r in old_records]

    sanitized = _sanitize(all_vision)
    deduped = _dedup(sanitized)

    logger.info(
        "After sanitize+dedup: %d records (was %d, diff=%+d)",
        len(deduped),
        len(old_records),
        len(deduped) - len(old_records),
    )

    # Count codes fixed vs dropped per province (for summary)
    codes_fixed: dict[str, int] = {}
    codes_dropped_from_old: dict[str, int] = {}
    for p in ("ON", "BC", "YT"):
        old_codes_set = {r.fsc_code for r in old_records if r.province == p}
        new_codes_set = {r.fsc_code for r in deduped if r.province == p}
        sanitized_p_set = {r.fsc_code for r in sanitized if r.province == p}
        # "fixed" means old code was bad but is now well-formed in new set
        fixed = sum(
            1 for r in old_records
            if r.province == p
            and r.fsc_code not in new_codes_set
            and (r.fsc_code.lstrip("#* ") != r.fsc_code or " " in r.fsc_code.strip())
        )
        dropped = len(old_codes_set) - len(
            {r.fsc_code for r in old_records if r.province == p} & sanitized_p_set
        )
        codes_fixed[p] = fixed
        codes_dropped_from_old[p] = dropped

    # ------------------------------------------------------------------
    # 3. Promote back to FeeCodeRecord, carrying over NGS where unchanged
    # ------------------------------------------------------------------
    logger.info("Promoting VisionRecords -> FeeCodeRecords with NGS carryover...")
    need_mapping: list[FeeCodeRecord] = []
    keep_ngs: list[FeeCodeRecord] = []

    for vr in deduped:
        pdf_hash = old_source_pdf_hashes.get(vr.province, "")
        promoted = _promote(vr, source_pdf_hash=pdf_hash)
        old_r = old_index.get((vr.province, vr.fsc_code))
        if old_r is not None and old_r.NGS_code is not None:
            # Code is unchanged and had a valid NGS assignment -- carry it over
            promoted = promoted.model_copy(update={
                "NGS_code": old_r.NGS_code,
                "NGS_label": old_r.NGS_label,
                "NGS_mapping_method": old_r.NGS_mapping_method,
                "NGS_mapping_confidence": old_r.NGS_mapping_confidence,
            })
            keep_ngs.append(promoted)
        else:
            # New code (from split/strip) or previously unmapped -- needs remapping
            need_mapping.append(promoted)

    logger.info(
        "NGS carryover: %d kept, %d need fresh mapping",
        len(keep_ngs),
        len(need_mapping),
    )

    # ------------------------------------------------------------------
    # 4. Load NGS reference data
    # ------------------------------------------------------------------
    ngs_records = []
    for docx_path in sorted(DOCX_DIR.glob("*.docx")):
        batch = parse_ngs_docx(docx_path)
        ngs_records.extend(batch)
        logger.info("  loaded %d NGS entries from %s", len(batch), docx_path.name)
    logger.info("Total NGS categories: %d", len(ngs_records))

    # ------------------------------------------------------------------
    # 5. OpenAI client + partial NGS remapping
    # ------------------------------------------------------------------
    client = OpenAIClient()

    if need_mapping:
        logger.info("Running NGS mapping for %d records...", len(need_mapping))
        remapped = map_ngs(
            need_mapping,
            ngs_records,
            client=client,
            embed_model=EMBED_MODEL,
            llm_model=LLM_MODEL,
            dim=EMBED_DIM,
        )
    else:
        remapped = []
        logger.info("No records need NGS remapping.")

    # Merge: keep_ngs already has NGS populated; remapped has fresh assignments
    # Maintain stable sort order (province, fsc_code)
    all_records: list[FeeCodeRecord] = keep_ngs + remapped
    all_records.sort(key=lambda r: (r.province, r.fsc_code))

    # ------------------------------------------------------------------
    # 6. Build embeddings (full rebuild -- record set has changed)
    # ------------------------------------------------------------------
    logger.info(
        "Building embeddings for %d records (%s @ %d dim)...",
        len(all_records),
        EMBED_MODEL,
        EMBED_DIM,
    )
    emb_arr, record_ids = build_embeddings(all_records, client=client, model=EMBED_MODEL, dim=EMBED_DIM)

    # ------------------------------------------------------------------
    # 7. Write outputs
    # ------------------------------------------------------------------
    DST_DIR.mkdir(parents=True, exist_ok=True)

    codes_path = DST_DIR / "codes.json"
    embeddings_path = DST_DIR / "embeddings.npz"
    manifest_path = DST_DIR / "manifest.json"

    codes_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in all_records], indent=2),
        encoding="utf-8",
    )

    save_npz(embeddings_path, emb_arr, record_ids)

    row_counts = {
        p: sum(1 for r in all_records if r.province == p) for p in ("ON", "BC", "YT")
    }
    manifest = Manifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        row_counts=row_counts,
        source_pdf_hashes=old_source_pdf_hashes,
        models={"embed": EMBED_MODEL, "extract": LLM_MODEL},
        regression_override="post-processing migration from v2026-04-15-vision",
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 8. Before/after summary
    # ------------------------------------------------------------------
    new_ngs = _count_ngs(all_records)
    old_ngs = _count_ngs(old_records)

    print()
    print("=" * 68)
    print("  BEFORE / AFTER MIGRATION SUMMARY")
    print("=" * 68)
    print(f"  Source artifact : {SRC_VERSION}")
    print(f"  Output artifact : {DST_VERSION}")
    print()
    print(f"  {'Province':<10}  {'Before':>8}  {'After':>8}  {'Fixed':>8}  {'Dropped':>9}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
    for p in ("ON", "BC", "YT"):
        before = sum(1 for r in old_records if r.province == p)
        after = sum(1 for r in all_records if r.province == p)
        print(
            f"  {p:<10}  {before:>8}  {after:>8}"
            f"  {codes_fixed.get(p, 0):>8}  {codes_dropped_from_old.get(p, 0):>9}"
        )
    before_total = len(old_records)
    after_total = len(all_records)
    print(f"  {'TOTAL':<10}  {before_total:>8}  {after_total:>8}")
    print()
    print(f"  NGS mapped before  : {old_ngs:>6} / {before_total} "
          f"({old_ngs / before_total * 100:.1f}%)" if before_total else "  NGS mapped before: N/A")
    print(f"  NGS mapped after   : {new_ngs:>6} / {after_total} "
          f"({new_ngs / after_total * 100:.1f}%)" if after_total else "  NGS mapped after: N/A")
    print()
    costs = client.costs.snapshot()
    total_tokens = sum(s["prompt_tokens"] + s["completion_tokens"] for s in costs.values())
    print(f"  OpenAI tokens used : {total_tokens:,}")
    print(f"  Output written to  : {DST_DIR}")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
