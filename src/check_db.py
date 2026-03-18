from pathlib import Path
import sqlite3
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "mapping.db"

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH}")

    code = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if code:
        rows = cur.execute("""
            SELECT province, page, fsc_code, code_occurrence, title, price,
                   section, subsection, quality_flag
            FROM fee_codes
            WHERE province = 'ON' AND fsc_code = ?
            ORDER BY code_occurrence
        """, (code,)).fetchall()
    else:
        rows = cur.execute("""
            SELECT province, page, fsc_code, code_occurrence, title, price,
                   section, subsection, quality_flag
            FROM fee_codes
            WHERE province = 'ON'
            ORDER BY page, fsc_code
            LIMIT 20
        """).fetchall()

    print(f"\nRows: {len(rows)}\n")
    for row in rows:
        print(dict(row))

    conn.close()

if __name__ == "__main__":
    main()