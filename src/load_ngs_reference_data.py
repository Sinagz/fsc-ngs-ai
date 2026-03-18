#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


CREATE_NGS_CATEGORIES_SQL = """
CREATE TABLE IF NOT EXISTS ngs_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ngs_code TEXT NOT NULL,
    ngs_label TEXT,
    ngs_description TEXT,
    section TEXT,
    source_file TEXT,
    source_section TEXT,
    raw_text TEXT
);
"""

CREATE_GROUPING_RULES_SQL = """
CREATE TABLE IF NOT EXISTS grouping_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    province TEXT,
    rule_type TEXT,
    trigger TEXT,
    outcome_ngs_code TEXT,
    notes TEXT,
    source_file TEXT,
    source_section TEXT,
    raw_text TEXT
);
"""

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ngs_categories_code ON ngs_categories(ngs_code);",
    "CREATE INDEX IF NOT EXISTS idx_grouping_rules_province ON grouping_rules(province);",
    "CREATE INDEX IF NOT EXISTS idx_grouping_rules_ngs ON grouping_rules(outcome_ngs_code);",
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load NGS reference JSONLs into SQLite.")
    parser.add_argument("--categories", default="data/parsed/ngs/ngs_categories.jsonl")
    parser.add_argument("--rules", default="data/parsed/ngs/grouping_rules.jsonl")
    parser.add_argument("--db", default="db/mapping.db")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    categories_path = Path(args.categories)
    rules_path = Path(args.rules)
    db_path = Path(args.db)

    if not categories_path.exists():
        raise FileNotFoundError(f"Categories file not found: {categories_path}")
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    categories = list(load_jsonl(categories_path))
    rules = list(load_jsonl(rules_path))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(CREATE_NGS_CATEGORIES_SQL)
    cur.execute(CREATE_GROUPING_RULES_SQL)
    for stmt in INDEX_SQL:
        cur.execute(stmt)

    if args.replace:
        cur.execute("DELETE FROM ngs_categories")
        cur.execute("DELETE FROM grouping_rules")

    for row in categories:
        cur.execute(
            """
            INSERT INTO ngs_categories (
                ngs_code, ngs_label, ngs_description, section,
                source_file, source_section, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("ngs_code"),
                row.get("ngs_label"),
                row.get("ngs_description"),
                row.get("section"),
                row.get("source_file"),
                row.get("source_section"),
                row.get("raw_text"),
            ),
        )

    for row in rules:
        cur.execute(
            """
            INSERT INTO grouping_rules (
                province, rule_type, trigger, outcome_ngs_code,
                notes, source_file, source_section, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("province"),
                row.get("rule_type"),
                row.get("trigger"),
                row.get("outcome_ngs_code"),
                row.get("notes"),
                row.get("source_file"),
                row.get("source_section"),
                row.get("raw_text"),
            ),
        )

    conn.commit()
    conn.close()

    print(f"Loaded {len(categories)} ngs_categories and {len(rules)} grouping_rules into {db_path}")


if __name__ == "__main__":
    main()