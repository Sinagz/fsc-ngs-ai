from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "mapping.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS fee_codes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        province TEXT NOT NULL,
        page INTEGER,
        fsc_code TEXT NOT NULL,
        code_occurrence INTEGER,
        title TEXT,
        price REAL,
        description TEXT,
        notes TEXT,
        section TEXT,
        subsection TEXT,
        entry_x0 REAL,
        entry_y0 REAL,
        font_size REAL,
        quality_flag TEXT,
        quality_reasons TEXT
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_fee_codes_province_code
    ON fee_codes (province, fsc_code)
    """)

    conn.commit()
    conn.close()

    print(f"Database initialized at: {DB_PATH}")

if __name__ == "__main__":
    main()