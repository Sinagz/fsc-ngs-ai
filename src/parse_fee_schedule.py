from pathlib import Path
import json
import csv
import re

BASE_DIR = Path(__file__).resolve().parent.parent
LAYOUT_JSON = BASE_DIR / "data" / "extracted" / "raw_layout" / "layout_lines.json"
PARSED_DIR = BASE_DIR / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = PARSED_DIR / "ontario_fee_codes.json"
OUTPUT_CSV = PARSED_DIR / "ontario_fee_codes.csv"

CODE_AT_START_RE = re.compile(r'^([A-Z]\d{3,4})\b')
PRICE_ONLY_RE = re.compile(r'^\$?(\d+\.\d{2})$')
PRICE_INLINE_RE = re.compile(r'(\d+\.\d{2})')
RULE_LINE_RE = re.compile(r'^\d+\.')
LETTER_RULE_RE = re.compile(r'^[a-z]\.')
DOT_LEADER_RE = re.compile(r'\.{5,}')
ADD_TRAIL_RE = re.compile(r'\badd\s*$', re.IGNORECASE)

SKIP_PAGES = set(range(1, 37))


def clean_text(text: str) -> str:
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_all_capsish(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.75


def normalize_heading(text: str) -> str:
    text = clean_text(text)
    return text


def clean_title_text(text: str) -> str:
    text = clean_text(text)
    text = DOT_LEADER_RE.sub(" ", text)
    text = ADD_TRAIL_RE.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip(" .-")
    return text


def looks_like_fee_entry(text: str, x0: float | None) -> bool:
    text = clean_text(text)

    if not text:
        return False
    if not CODE_AT_START_RE.match(text):
        return False
    if len(text.split()) < 2:
        return False
    if x0 is not None and not (35 <= x0 <= 55):
        return False
    if text.count(",") >= 3:
        return False

    return True


def parse_code_title_price(text: str):
    text = clean_text(text)
    m = CODE_AT_START_RE.match(text)
    code = m.group(1) if m else None
    remaining = text[m.end():].strip() if m else text

    price = None
    price_match = PRICE_INLINE_RE.search(remaining)
    if price_match:
        price = price_match.group(1)
        remaining = (remaining[:price_match.start()] + " " + remaining[price_match.end():]).strip()

    title = clean_title_text(remaining)
    return code, title, price


def is_probable_heading(text: str, x0: float | None) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if CODE_AT_START_RE.match(text):
        return False

    if is_all_capsish(text) and len(text.split()) <= 14:
        return True

    if x0 is not None and x0 <= 40 and is_all_capsish(text):
        return True

    return False


def is_price_only_line(text: str) -> bool:
    return bool(PRICE_ONLY_RE.match(clean_text(text)))


def is_rule_or_note_line(text: str) -> bool:
    text = clean_text(text)
    low = text.lower()

    if RULE_LINE_RE.match(text):
        return True
    if LETTER_RULE_RE.match(text):
        return True
    if low.startswith("paymentrules"):
        return True
    if low.startswith("payment rules"):
        return True
    if low.startswith("medicalrecordrequirements"):
        return True
    if low.startswith("medical record requirements"):
        return True
    if low.startswith("claimssubmissioninstructions"):
        return True
    if low.startswith("claims submission instructions"):
        return True
    if low.startswith("note"):
        return True
    if low.startswith("[commentary"):
        return True

    return False


def looks_like_wrapped_entry_continuation(text: str, x0: float | None) -> bool:
    text = clean_text(text)
    low = text.lower()

    if not text:
        return False
    if CODE_AT_START_RE.match(text):
        return False
    if is_probable_heading(text, x0):
        return False
    if is_rule_or_note_line(text):
        return False

    if x0 is not None and not (35 <= x0 <= 95):
        return False

    if "listed in payment rule" in low:
        return True
    if "per visit" in low:
        return True
    if "subject to same" in low:
        return True
    if "add " in low:
        return True

    return False


def append_text(existing: str, new_text: str) -> str:
    new_text = clean_text(new_text)
    if not new_text:
        return existing
    if not existing:
        return new_text
    return existing + " " + new_text


def should_close_on_heading(text: str, x0: float | None) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if CODE_AT_START_RE.match(text):
        return False
    return is_probable_heading(text, x0)


def main():
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"Missing file: {LAYOUT_JSON}")

    data = json.loads(LAYOUT_JSON.read_text(encoding="utf-8"))

    records = []
    current_record = None
    current_section = None
    current_subsection = None

    for page in data:
        page_num = page["page"]

        if page_num in SKIP_PAGES:
            continue

        lines = page["lines"]
        i = 0

        while i < len(lines):
            line = lines[i]
            text = clean_text(line.get("text", ""))
            x0 = line.get("x0")
            y0 = line.get("y0")
            font_size = line.get("font_size")

            if not text:
                i += 1
                continue

            # heading
            if is_probable_heading(text, x0):
                if current_record:
                    records.append(current_record)
                    current_record = None

                heading = normalize_heading(text)
                if x0 is not None and x0 <= 40:
                    current_section = heading
                    current_subsection = None
                else:
                    current_subsection = heading

                i += 1
                continue

            # new fee entry
            if looks_like_fee_entry(text, x0):
                if current_record:
                    records.append(current_record)

                code, title, price = parse_code_title_price(text)

                current_record = {
                    "province": "ON",
                    "page": page_num,
                    "fsc_code": code,
                    "title": title,
                    "price": price,
                    "description": "",
                    "notes": "",
                    "section": current_section,
                    "subsection": current_subsection,
                    "entry_x0": x0,
                    "entry_y0": y0,
                    "font_size": font_size,
                }

                # look-ahead: title continuation or next-line price
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    nt = clean_text(nxt.get("text", ""))
                    nx0 = nxt.get("x0")

                    if not nt:
                        j += 1
                        continue

                    if looks_like_fee_entry(nt, nx0):
                        break
                    if should_close_on_heading(nt, nx0):
                        break

                    if is_price_only_line(nt) and not current_record["price"]:
                        current_record["price"] = PRICE_ONLY_RE.match(nt).group(1)
                        j += 1
                        continue

                    if looks_like_wrapped_entry_continuation(nt, nx0):
                        # capture inline price if present
                        pm = PRICE_INLINE_RE.search(nt)
                        if pm and not current_record["price"]:
                            current_record["price"] = pm.group(1)
                            nt = (nt[:pm.start()] + " " + nt[pm.end():]).strip()

                        current_record["title"] = append_text(current_record["title"], clean_title_text(nt))
                        j += 1
                        continue

                    break

                i = j
                continue

            # attach body to current record
            if current_record:
                if should_close_on_heading(text, x0):
                    records.append(current_record)
                    current_record = None

                    heading = normalize_heading(text)
                    if x0 is not None and x0 <= 40:
                        current_section = heading
                        current_subsection = None
                    else:
                        current_subsection = heading

                    i += 1
                    continue

                if is_price_only_line(text) and not current_record["price"]:
                    current_record["price"] = PRICE_ONLY_RE.match(text).group(1)
                    i += 1
                    continue

                if is_rule_or_note_line(text) or (x0 is not None and x0 >= 60):
                    current_record["notes"] = append_text(current_record["notes"], text)
                else:
                    current_record["description"] = append_text(current_record["description"], text)

            i += 1

    if current_record:
        records.append(current_record)

    # add a duplicate counter per code
    seen = {}
    for r in records:
        code = r["fsc_code"]
        seen[code] = seen.get(code, 0) + 1
        r["code_occurrence"] = seen[code]

    OUTPUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "province",
                "page",
                "fsc_code",
                "code_occurrence",
                "title",
                "price",
                "description",
                "notes",
                "section",
                "subsection",
                "entry_x0",
                "entry_y0",
                "font_size",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Parsed {len(records)} fee code records.")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"CSV saved to:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()