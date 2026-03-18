from pathlib import Path
import csv
import re

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "parsed" / "ontario_fee_codes.csv"
OUTPUT_CSV = BASE_DIR / "data" / "parsed" / "ontario_fee_codes_classified.csv"

DOTS_RE = re.compile(r'\.{5,}')
BAD_SUBSECTION_RE = re.compile(r'GP\d+\)')
PRICE_RE = re.compile(r'\b\d+\.\d{2}\b')


def classify_row(row):
    score_bad = 0
    reasons = []

    title = (row.get("title") or "").strip()
    desc = (row.get("description") or "").strip()
    notes = (row.get("notes") or "").strip()
    subsection = (row.get("subsection") or "").strip()
    occurrence = int(row.get("code_occurrence") or 1)

    if DOTS_RE.search(title):
        score_bad += 2
        reasons.append("title_has_dot_leaders")

    if DOTS_RE.search(subsection):
        score_bad += 2
        reasons.append("subsection_has_dot_leaders")

    if BAD_SUBSECTION_RE.search(subsection):
        score_bad += 2
        reasons.append("subsection_looks_like_gp_reference")

    if len(desc) > 200 and PRICE_RE.search(desc):
        score_bad += 2
        reasons.append("description_contains_embedded_prices")

    if len(notes) > 500 and "see " in notes.lower():
        score_bad += 1
        reasons.append("very_long_reference_like_notes")

    if occurrence > 1:
        reasons.append("duplicate_code_occurrence")

    if score_bad >= 4:
        quality = "likely_reference_or_noisy"
    elif occurrence > 1:
        quality = "probably_good_duplicate"
    else:
        quality = "good"

    return quality, ";".join(reasons)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output_rows = []
    for row in rows:
        quality, reasons = classify_row(row)
        row["quality_flag"] = quality
        row["quality_reasons"] = reasons
        output_rows.append(row)

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Classified {len(output_rows)} rows.")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()