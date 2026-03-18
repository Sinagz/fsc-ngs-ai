#!/usr/bin/env python3
import argparse
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "per", "each", "visit", "service", "services", "procedure", "procedures",
    "by", "at", "from", "as", "is", "are", "be", "this", "that", "full",
    "quarter", "hour", "payment", "rules", "including"
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]
    return tokens


def jaccard_score(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_score(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def combined_score(fee_text: str, ngs_text: str) -> Tuple[float, List[str]]:
    fee_tokens = tokenize(fee_text)
    ngs_tokens = tokenize(ngs_text)

    shared = sorted(set(fee_tokens) & set(ngs_tokens))
    j = jaccard_score(fee_tokens, ngs_tokens)
    o = overlap_score(fee_tokens, ngs_tokens)

    score = (0.4 * j) + (0.6 * o)
    return score, shared


def confidence_from_score(score: float) -> str:
    if score >= 0.60:
        return "high"
    if score >= 0.35:
        return "medium"
    if score >= 0.18:
        return "low"
    return "very_low"


def fetch_fee_codes(conn: sqlite3.Connection) -> List[Dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT province, fsc_code, title, notes
        FROM fee_codes
        WHERE province = 'ON'
          AND quality_flag IN ('good', 'probably_good_duplicate')
        ORDER BY fsc_code
        """
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_ngs_categories(conn: sqlite3.Connection) -> List[Dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ngs_code, ngs_label, ngs_description
        FROM ngs_categories
        ORDER BY ngs_code
        """
    )
    return [dict(r) for r in cur.fetchall()]


def create_output_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inferred_ngs_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            province TEXT NOT NULL,
            fsc_code TEXT NOT NULL,
            fee_title TEXT,
            fee_notes TEXT,
            ngs_code TEXT,
            ngs_label TEXT,
            ngs_description TEXT,
            match_method TEXT,
            score REAL,
            confidence TEXT,
            rationale TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_inferred_candidates_fsc ON inferred_ngs_candidates(province, fsc_code)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_inferred_candidates_ngs ON inferred_ngs_candidates(ngs_code)"
    )
    conn.commit()


def clear_existing_for_province(conn: sqlite3.Connection, province: str) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM inferred_ngs_candidates WHERE province = ?", (province,))
    conn.commit()


def infer_candidates_for_fee(fee_row: Dict, ngs_rows: List[Dict], top_n: int = 5) -> List[Dict]:
    fee_title = fee_row.get("title") or ""
    fee_notes = fee_row.get("notes") or ""
    fee_text = f"{fee_title} {fee_notes}".strip()

    scored = []
    for ngs in ngs_rows:
        ngs_text = f"{ngs.get('ngs_label') or ''} {ngs.get('ngs_description') or ''}".strip()
        score, shared = combined_score(fee_text, ngs_text)

        if score <= 0:
            continue

        rationale = f"Shared terms: {', '.join(shared[:10])}" if shared else "Weak lexical overlap"
        scored.append({
            "province": fee_row["province"],
            "fsc_code": fee_row["fsc_code"],
            "fee_title": fee_title,
            "fee_notes": fee_notes,
            "ngs_code": ngs["ngs_code"],
            "ngs_label": ngs["ngs_label"],
            "ngs_description": ngs["ngs_description"],
            "match_method": "text_similarity_baseline",
            "score": round(score, 4),
            "confidence": confidence_from_score(score),
            "rationale": rationale,
        })

    scored.sort(key=lambda x: (-x["score"], x["ngs_code"]))
    return scored[:top_n]


def insert_candidates(conn: sqlite3.Connection, rows: List[Dict]) -> int:
    cur = conn.cursor()
    inserted = 0
    for row in rows:
        cur.execute(
            """
            INSERT INTO inferred_ngs_candidates (
                province, fsc_code, fee_title, fee_notes,
                ngs_code, ngs_label, ngs_description,
                match_method, score, confidence, rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["province"],
                row["fsc_code"],
                row["fee_title"],
                row["fee_notes"],
                row["ngs_code"],
                row["ngs_label"],
                row["ngs_description"],
                row["match_method"],
                row["score"],
                row["confidence"],
                row["rationale"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer baseline Ontario -> NGS candidates.")
    parser.add_argument("--db", default="db/mapping.db", help="Path to SQLite database")
    parser.add_argument("--top-n", type=int, default=5, help="Top N candidates per fee code")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for testing")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    create_output_table(conn)
    clear_existing_for_province(conn, "ON")

    fee_rows = fetch_fee_codes(conn)
    ngs_rows = fetch_ngs_categories(conn)

    if args.limit > 0:
        fee_rows = fee_rows[:args.limit]

    total_inserted = 0
    for fee in fee_rows:
        candidates = infer_candidates_for_fee(fee, ngs_rows, top_n=args.top_n)
        total_inserted += insert_candidates(conn, candidates)

    conn.close()
    print(f"Inserted {total_inserted} inferred candidate rows for Ontario.")


if __name__ == "__main__":
    main()