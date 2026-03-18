#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from docx import Document

NGS_LINE_RE = re.compile(r"^\s*(\d{3}\.\d{3})\s+(.+?)\s*$")
SECTION_LIKE_RE = re.compile(r"^[A-Z][A-Z\s/&\-\(\)]+$")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_section_heading(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if NGS_LINE_RE.match(t):
        return False
    if len(t) > 100:
        return False
    return bool(SECTION_LIKE_RE.match(t))


def parse_docx(docx_path: Path) -> List[Dict]:
    doc = Document(str(docx_path))
    source_file = docx_path.name

    paragraphs = [normalize_text(p.text) for p in doc.paragraphs]
    paragraphs = [p for p in paragraphs if p]

    rows: List[Dict] = []
    current_section = ""
    current_category = None

    for idx, text in enumerate(paragraphs, start=1):
        m = NGS_LINE_RE.match(text)

        if looks_like_section_heading(text):
            current_section = text
            continue

        if m:
            if current_category:
                rows.append(current_category)

            ngs_code = m.group(1)
            ngs_label = normalize_text(m.group(2))

            current_category = {
                "ngs_code": ngs_code,
                "ngs_label": ngs_label,
                "ngs_description": "",
                "section": current_section,
                "source_file": source_file,
                "source_section": f"paragraph_{idx}",
                "raw_text": text,
            }
            continue

        # attach descriptive text to the most recent NGS row
        if current_category:
            current_category["ngs_description"] = normalize_text(
                (current_category["ngs_description"] + " " + text).strip()
            )
            current_category["raw_text"] = normalize_text(
                current_category["raw_text"] + " || " + text
            )

    if current_category:
        rows.append(current_category)

    # dedupe
    deduped = []
    seen = set()
    for row in rows:
        key = (row["ngs_code"], row["ngs_label"], row["ngs_description"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NGS category DOCX into JSONL.")
    parser.add_argument(
        "--input",
        default="data/raw/docx/NPDB National Grouping System Categories.docx",
        help="Path to NPDB National Grouping System Categories.docx",
    )
    parser.add_argument(
        "--output",
        default="data/parsed/ngs/ngs_categories.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = parse_docx(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} NGS categories to {output_path}")


if __name__ == "__main__":
    main()