from pathlib import Path
import csv
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "parsed" / "ontario_fee_codes_classified.csv"
DB_PATH = BASE_DIR / "db" / "mapping.db"

KEEP_FLAGS = {"good", "probably_good_duplicate"}

def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def to_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH}")

    with CSV_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("quality_flag") in KEEP_FLAGS]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DELETE FROM fee_codes WHERE province = 'ON'")

    for row in rows:
        cur.execute("""
            INSERT INTO fee_codes (
                province, page, fsc_code, code_occurrence, title, price,
                description, notes, section, subsection,
                entry_x0, entry_y0, font_size,
                quality_flag, quality_reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("province"),
            to_int(row.get("page")),
            row.get("fsc_code"),
            to_int(row.get("code_occurrence")),
            row.get("title"),
            to_float(row.get("price")),
            row.get("description"),
            row.get("notes"),
            row.get("section"),
            row.get("subsection"),
            to_float(row.get("entry_x0")),
            to_float(row.get("entry_y0")),
            to_float(row.get("font_size")),
            row.get("quality_flag"),
            row.get("quality_reasons"),
        ))

    conn.commit()

    count = cur.execute("SELECT COUNT(*) FROM fee_codes WHERE province = 'ON'").fetchone()[0]
    conn.close()

    print(f"Loaded {count} Ontario fee code rows into {DB_PATH}")

if __name__ == "__main__":
    main()