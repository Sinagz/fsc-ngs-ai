"""
Re-extract zero-code windows from existing OCR cache.

Identifies windows in the extraction checkpoint that produced 0 codes but
have substantial OCR content (likely API timeouts). Re-runs extraction for
those windows only and merges results back.

Usage:
    python scripts/reextract_zeros.py [--province ON|BC|YT] [--min-code-hints 10]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from extract_mistral import (
    _call_with_retry, _extract_codes, STEP, WINDOW,
    EXTRACTION_PROMPT, EXTRACTION_SYSTEM, CHAT_MODEL,
)

from mistralai.client import Mistral

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "data" / "extracted" / "mistral"


def reextract_province(province: str, client: Mistral, min_code_hints: int):
    prov = province.lower()
    ocr_path   = OUT_DIR / f"{prov}_ocr.json"
    ckpt_path  = OUT_DIR / f"{prov}_extract.ckpt.json"
    codes_path = OUT_DIR / f"{prov}_codes.json"

    if not ocr_path.exists():
        print(f"[{province}] OCR cache not found: {ocr_path.name} — skipping")
        return
    if not ckpt_path.exists():
        print(f"[{province}] Extraction checkpoint not found — skipping")
        return

    print(f"[{province}] Loading OCR cache...")
    ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
    page_md: dict[int, str] = {p["index"]: p["markdown"] for p in ocr_data}

    print(f"[{province}] Loading extraction checkpoint...")
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    all_raw: list[dict] = ckpt.get("raw", [])
    done_windows: set[int] = set(ckpt.get("done_windows", []))

    # Count codes per window start
    window_codes: dict[int, int] = {}
    for c in all_raw:
        w = c.get("page_start")
        if w is not None:
            window_codes[w] = window_codes.get(w, 0) + 1

    # Identify zero-code windows with significant OCR content
    zero_windows = sorted([w for w in done_windows if window_codes.get(w, 0) == 0])

    candidates = []
    for start in zero_windows:
        end = start + WINDOW - 1
        pages = list(range(start, end + 1))
        combined = "\n".join(page_md.get(p, "") for p in pages)
        code_hints = len(re.findall(r"\b[A-Z]\d{4,}\b|\b\d{5}\b|\bV\d{5}\b|\b[A-Z]\d{3}\b", combined))
        if code_hints >= min_code_hints:
            candidates.append((start, end, code_hints))

    print(f"[{province}] {len(zero_windows)} zero-code windows, {len(candidates)} with >= {min_code_hints} code hints:")
    for start, end, hints in candidates:
        print(f"  window {start}-{end}: {hints} code hints")

    if not candidates:
        print(f"[{province}] Nothing to re-extract.")
        return

    # Re-extract each candidate window
    newly_added = 0
    for start, end, hints in candidates:
        pages = list(range(start, end + 1))
        combined_md = "\n\n---PAGE BREAK---\n\n".join(page_md.get(p, "") for p in pages)

        print(f"  [{province}] Re-extracting window {start}-{end} ({hints} code hints)...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = _extract_codes(client, combined_md, start, end)
        except Exception as e:
            print(f"\n    [SKIP] {str(e)[:80]}")
            continue

        codes   = result.get("codes", [])
        chapter = result.get("chapter", "")
        section = result.get("section", "")

        for c in codes:
            c["chapter"]    = c.get("chapter") or chapter
            c["section"]    = c.get("section") or section
            c["page_start"] = start
            c["province"]   = province

        all_raw.extend(codes)
        newly_added += len(codes)
        print(f"({time.time()-t0:.1f}s, {len(codes)} codes)")
        time.sleep(0.3)

    print(f"[{province}] Added {newly_added} new codes from {len(candidates)} windows")

    # Save updated checkpoint
    ckpt_path.write_text(
        json.dumps({"raw": all_raw, "done_windows": list(done_windows)}, ensure_ascii=False),
        encoding="utf-8"
    )

    # Re-deduplicate and save codes
    seen: dict[str, dict] = {}
    for entry in all_raw:
        code = (entry.get("fsc_code") or "").strip().upper()
        if not code:
            continue
        if code not in seen:
            entry["fsc_code"] = code
            seen[code] = entry

    deduped = list(seen.values())
    codes_path.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{province}] {len(all_raw)} raw -> {len(deduped)} unique codes -> {codes_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Re-extract zero-code windows from OCR cache")
    parser.add_argument("--province", choices=["ON", "BC", "YT"],
                        help="Run for a single province (default: all)")
    parser.add_argument("--min-code-hints", type=int, default=10,
                        help="Min code-like patterns to qualify for re-extraction (default: 10)")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY not set")

    client = Mistral(api_key=api_key, timeout_ms=300_000)

    provinces = [args.province] if args.province else ["ON", "BC", "YT"]
    for prov in provinces:
        reextract_province(prov, client, args.min_code_hints)


if __name__ == "__main__":
    main()
