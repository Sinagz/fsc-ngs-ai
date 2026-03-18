from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).resolve().parent.parent
LAYOUT_JSON = BASE_DIR / "data" / "extracted" / "raw_layout" / "layout_lines.json"

FEE_CODE_PATTERN = re.compile(r"\b([A-Z]\d{3,4})\b")


def main():
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"Missing file: {LAYOUT_JSON}")

    data = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))

    matches = []

    for page in data:
        page_num = page["page"]
        for line in page["lines"]:
            text = line.get("text", "").strip()
            if FEE_CODE_PATTERN.search(text):
                matches.append({
                    "page": page_num,
                    "x0": line.get("x0"),
                    "y0": line.get("y0"),
                    "font_size": line.get("font_size"),
                    "text": text
                })

    print(f"\nFound {len(matches)} lines with fee-code-like patterns.\n")

    for item in matches[:300]:
        print(
            f"page={item['page']:>3} | x0={item['x0']:>7} | y0={item['y0']:>7} | "
            f"size={item['font_size']:>5} | {item['text']}"
        )


if __name__ == "__main__":
    main()