#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from docx import Document

PROVINCES = [
    "Newfoundland",
    "Prince Edward Island",
    "Nova Scotia",
    "New Brunswick",
    "Québec",
    "Ontario",
    "Manitoba",
    "Saskatchewan",
    "Alberta",
    "British Columbia",
    "Yukon",
]

NGS_CODE_RE = re.compile(r"NGS\s+(\d{3}\.\d{3})", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def province_code(name: str) -> str:
    mapping = {
        "Ontario": "ON",
        "British Columbia": "BC",
        "Yukon": "YT",
        "Alberta": "AB",
        "Saskatchewan": "SK",
        "Manitoba": "MB",
        "Québec": "QC",
        "New Brunswick": "NB",
        "Nova Scotia": "NS",
        "Prince Edward Island": "PE",
        "Newfoundland": "NL",
    }
    return mapping.get(name, name[:2].upper())


def parse_rules(docx_path: Path) -> List[Dict]:
    doc = Document(str(docx_path))
    source_file = docx_path.name
    paragraphs = [normalize_text(p.text) for p in doc.paragraphs]
    paragraphs = [p for p in paragraphs if p]

    rows: List[Dict] = []
    current_province = None

    for idx, text in enumerate(paragraphs, start=1):
        if text in PROVINCES:
            current_province = text
            continue

        # global rules
        if "surgical assistance" in text.lower() and "NGS 073.000" in text:
            rows.append({
                "province": "ALL",
                "rule_type": "role_code_rule",
                "trigger": "role code indicates surgical assistance",
                "outcome_ngs_code": "073.000",
                "notes": text,
                "source_file": source_file,
                "source_section": f"paragraph_{idx}",
                "raw_text": text,
            })

        if "anesthesia" in text.lower() and "NGS 075.000" in text:
            rows.append({
                "province": "ALL",
                "rule_type": "role_code_rule",
                "trigger": "role code indicates anesthesia",
                "outcome_ngs_code": "075.000",
                "notes": text,
                "source_file": source_file,
                "source_section": f"paragraph_{idx}",
                "raw_text": text,
            })

        # ON role codes
        if current_province == "Ontario":
            if text == "B":
                rows.append({
                    "province": "ON",
                    "rule_type": "role_code_value",
                    "trigger": "role code B",
                    "outcome_ngs_code": "073.000",
                    "notes": "Assistant's Service",
                    "source_file": source_file,
                    "source_section": f"paragraph_{idx}",
                    "raw_text": text,
                })
            elif text == "C":
                rows.append({
                    "province": "ON",
                    "rule_type": "role_code_value",
                    "trigger": "role code C",
                    "outcome_ngs_code": "075.000",
                    "notes": "Anesthetist's Service",
                    "source_file": source_file,
                    "source_section": f"paragraph_{idx}",
                    "raw_text": text,
                })

            if "The first four characters of the FSC code correspond to the fee code" in text:
                rows.append({
                    "province": "ON",
                    "rule_type": "fsc_structure",
                    "trigger": "first 4 chars of FSC",
                    "outcome_ngs_code": None,
                    "notes": text,
                    "source_file": source_file,
                    "source_section": f"paragraph_{idx}",
                    "raw_text": text,
                })

        # BC structure
        if current_province == "British Columbia":
            if "every fee code is padded with two leading zeros" in text.lower():
                rows.append({
                    "province": "BC",
                    "rule_type": "fsc_structure",
                    "trigger": "drop first 2 leading zeros from FSC",
                    "outcome_ngs_code": None,
                    "notes": text,
                    "source_file": source_file,
                    "source_section": f"paragraph_{idx}",
                    "raw_text": text,
                })

        # Yukon structure
        if current_province == "Yukon":
            if "If the FSC code is 4 digits" in text or "If the FSC code is 6-digit" in text:
                rows.append({
                    "province": "YT",
                    "rule_type": "fsc_structure",
                    "trigger": "Yukon FSC parsing",
                    "outcome_ngs_code": None,
                    "notes": text,
                    "source_file": source_file,
                    "source_section": f"paragraph_{idx}",
                    "raw_text": text,
                })

    # dedupe
    deduped = []
    seen = set()
    for row in rows:
        key = (
            row["province"],
            row["rule_type"],
            row["trigger"],
            row["outcome_ngs_code"],
            row["notes"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse grouping rules DOCX into JSONL.")
    parser.add_argument(
        "--input",
        default="data/raw/docx/Fee Codes and Grouping Rules.docx",
        help="Path to Fee Codes and Grouping Rules.docx",
    )
    parser.add_argument(
        "--output",
        default="data/parsed/ngs/grouping_rules.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = parse_rules(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} grouping rules to {output_path}")


if __name__ == "__main__":
    main()