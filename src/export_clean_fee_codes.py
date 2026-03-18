from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "parsed" / "ontario_fee_codes_classified.csv"
OUTPUT_CSV = BASE_DIR / "data" / "parsed" / "ontario_fee_codes_clean.csv"

KEEP_FLAGS = {"good", "probably_good_duplicate"}

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    clean_rows = [r for r in rows if r.get("quality_flag") in KEEP_FLAGS]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean_rows)

    print(f"Exported {len(clean_rows)} clean rows.")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()