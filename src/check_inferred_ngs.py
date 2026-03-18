#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check inferred NGS candidates for a fee code.")
    parser.add_argument("province", help="Province code, e.g. ON")
    parser.add_argument("fsc_code", help="FSC code, e.g. K040")
    parser.add_argument("--db", default="db/mapping.db", help="Path to SQLite DB")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            province, fsc_code, fee_title,
            ngs_code, ngs_label,
            score, confidence, match_method, rationale
        FROM inferred_ngs_candidates
        WHERE province = ?
          AND fsc_code = ?
        ORDER BY score DESC, ngs_code
        """,
        (args.province.upper().strip(), args.fsc_code.upper().strip()),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"No inferred candidates found for {args.province.upper()} {args.fsc_code.upper()}")
        return

    print(f"Inferred NGS candidates for {args.province.upper()} {args.fsc_code.upper()}")
    print("=" * 100)
    for i, row in enumerate(rows, start=1):
        print(f"[{i}]")
        print(f"Province    : {row['province']}")
        print(f"FSC code    : {row['fsc_code']}")
        print(f"Fee title   : {row['fee_title']}")
        print(f"NGS code    : {row['ngs_code']}")
        print(f"NGS label   : {row['ngs_label']}")
        print(f"Score       : {row['score']}")
        print(f"Confidence  : {row['confidence']}")
        print(f"Method      : {row['match_method']}")
        print(f"Rationale   : {row['rationale']}")
        print("-" * 100)


if __name__ == "__main__":
    main()