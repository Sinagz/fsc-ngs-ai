from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).resolve().parent.parent
LAYOUT_JSON = BASE_DIR / "data" / "extracted" / "raw_layout" / "layout_lines.json"

CODE_RE = re.compile(r'^[A-Z]\d{3,4}\b')
ANY_CODE_RE = re.compile(r'\b[A-Z]\d{3,4}\b')
PRICE_RE = re.compile(r'\b\d+\.\d{2}\b')

# things that usually mean "not a real entry row"
BAD_PREFIXES = (
    "Claims for",
    "use the",
    "see ",
    "When ",
    "While ",
    "A psychiatric assessment",
)

def looks_like_real_entry(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    # must START with a code, not just contain one somewhere
    if not CODE_RE.match(text):
        return False

    # reject code-only lines like "A102"
    parts = text.split()
    if len(parts) < 2:
        return False

    # reject lines that are mostly code lists
    if "," in text and text.count(",") >= 2:
        return False

    # reject lines that start like numbered notes after code references
    if text.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
        return False

    # reject obvious narrative/reference lines
    for prefix in BAD_PREFIXES:
        if text.startswith(prefix):
            return False

    # reject lines that look like "(C123)" only or mostly reference text
    if text.startswith("(") and text.endswith(")"):
        return False

    # keep if code + title words exist
    return True


def main():
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"Missing file: {LAYOUT_JSON}")

    data = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))

    candidates = []

    for page in data:
        page_num = page["page"]
        for line in page["lines"]:
            text = line.get("text", "").strip()
            if looks_like_real_entry(text):
                candidates.append({
                    "page": page_num,
                    "x0": line.get("x0"),
                    "y0": line.get("y0"),
                    "font_size": line.get("font_size"),
                    "text": text
                })

    print(f"\nFound {len(candidates)} likely fee entry candidate lines.\n")

    for item in candidates[:400]:
        print(
            f"page={item['page']:>3} | x0={item['x0']:>7} | y0={item['y0']:>7} | "
            f"size={item['font_size']:>5} | {item['text']}"
        )


if __name__ == "__main__":
    main()