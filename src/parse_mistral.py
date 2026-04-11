"""
Convert Mistral-extracted fee codes into the standard all_fee_codes.json format
so the rest of the pipeline (map_fsc_ngs, build_embeddings, etc.) is unchanged.

Input:  data/extracted/mistral/{on,bc,yt}_codes.json
Output: data/parsed/fee_codes/all_fee_codes.json  (same schema as before)

Usage:
    python src/parse_mistral.py
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
MISTRAL_DIR = BASE_DIR / "data" / "extracted" / "mistral"
OUT_DIR     = BASE_DIR / "data" / "parsed" / "fee_codes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVINCES = ["ON", "BC", "YT"]


def _normalize(entry: dict) -> dict:
    """Map Mistral extraction fields to the canonical schema."""
    price = str(entry.get("price") or "").replace("$", "").strip()
    return {
        "province":        entry.get("province", ""),
        "fsc_code":        (entry.get("fsc_code") or "").strip().upper(),
        "fsc_fn":          (entry.get("fsc_fn") or "").strip(),
        "fsc_chapter":     (entry.get("chapter") or "").strip(),
        "fsc_section":     (entry.get("section") or "").strip(),
        "fsc_subsection":  "",
        "fsc_description": (entry.get("fsc_description") or "").strip(),
        "fsc_notes":       "",
        "fsc_others":      "",
        "price":           price,
        "page":            entry.get("page_start", ""),
        "fsc_rationale":   "",
        "fsc_confidence":  "",
        "fsc_key_observations": "",
    }


def main():
    all_codes: list[dict] = []

    for prov in PROVINCES:
        codes_path = MISTRAL_DIR / f"{prov.lower()}_codes.json"
        if not codes_path.exists():
            print(f"[{prov}] {codes_path.name} not found — skipping")
            continue

        raw = json.loads(codes_path.read_text(encoding="utf-8"))
        normalized = [_normalize(e) for e in raw if e.get("fsc_code")]
        all_codes.extend(normalized)
        print(f"[{prov}] {len(normalized)} codes")

    out_path = OUT_DIR / "all_fee_codes.json"
    out_path.write_text(json.dumps(all_codes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal: {len(all_codes)} codes -> {out_path}")


if __name__ == "__main__":
    main()
