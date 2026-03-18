from pathlib import Path
import json
from collections import Counter
import re

BASE_DIR = Path(__file__).resolve().parent.parent
LAYOUT_JSON = BASE_DIR / "data" / "extracted" / "raw_layout" / "layout_lines.json"

CODE_ONLY_RE = re.compile(r'^[A-Z]\d{3,4}$')
ENTRY_RE = re.compile(r'^[A-Z]\d{3,4}\s+\S+')


def main():
    data = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))

    print("\n=== PAGE PROFILE ===\n")

    for page in data:
        page_num = page["page"]
        code_only = 0
        entry_like = 0
        total = 0

        for line in page["lines"]:
            text = line.get("text", "").strip()
            if not text:
                continue
            total += 1

            if CODE_ONLY_RE.match(text):
                code_only += 1
            if ENTRY_RE.match(text):
                entry_like += 1

        print(
            f"page={page_num:>3} | total_lines={total:>4} | "
            f"code_only={code_only:>3} | entry_like={entry_like:>3}"
        )


if __name__ == "__main__":
    main()