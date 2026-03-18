from pathlib import Path
import csv
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "parsed" / "ontario_fee_codes_classified.csv"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing: {CSV_PATH}")

    target_code = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None

    with CSV_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if target_code:
        rows = [r for r in rows if r["fsc_code"] == target_code]

    print(f"\nRows shown: {len(rows)}\n")

    for row in rows:
        print("=" * 100)
        print(f"PAGE: {row['page']}")
        print(f"CODE: {row['fsc_code']}")
        print(f"OCCURRENCE: {row['code_occurrence']}")
        print(f"QUALITY: {row['quality_flag']}")
        print(f"REASONS: {row['quality_reasons']}")
        print(f"TITLE: {row['title']}")
        print(f"PRICE: {row['price']}")
        print(f"SECTION: {row['section']}")
        print(f"SUBSECTION: {row['subsection']}")
        print(f"DESCRIPTION: {row['description'][:300]}")
        print(f"NOTES: {row['notes'][:300]}")

if __name__ == "__main__":
    main()