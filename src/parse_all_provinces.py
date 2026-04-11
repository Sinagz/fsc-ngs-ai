"""
Parse fee codes from all 3 province PDFs (v2 – corrected parsers).

Key layout facts (from analysis):
  ON:  code [A-Z]\d{3,4} at x0~43, sz=12;  running head at x0~170;  price on
       separate line x0>350;  real content starts ~pg 130.
  BC:  code at x0~72 sz~10;  description on NEXT line x0~126 sz~10;
       notes x0~126-151 sz~9;  section heading x0~72 sz>10.3.
  YT:  code \d{4-6} at x0~67-72 sz~9;  description on NEXT line x0~110;
       price x0~490;  chapter heading sz=16;  subcategory x0~40 sz~9.
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LAYOUT_DIR = BASE_DIR / "data" / "extracted" / "raw_layout"
OUT_DIR = BASE_DIR / "data" / "parsed" / "fee_codes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── shared helpers ────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    for old, new in [("\u2013", "-"), ("\u2014", "-"), ("\u2019", "'"),
                     ("\u00a0", " "), ("\u00ae", ""), ("\ufffd", ""),
                     ("\u2022", ""), ("\u2018", "'")]:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()

def strip_dots(text: str) -> str:
    return re.sub(r"\.{3,}", " ", text).strip(" .-")

PRICE_INLINE_RE = re.compile(r"\$?(\d{1,6}\.\d{2})\s*$")
PRICE_ONLY_RE   = re.compile(r"^\$?(\d{1,6}\.\d{2})$")

def split_price(text: str):
    """Return (text_no_price, price_str_or_None)."""
    m = PRICE_INLINE_RE.search(text)
    if m:
        return strip_dots(text[: m.start()]), m.group(1)
    return strip_dots(text), None


# ── ONTARIO ──────────────────────────────────────────────────────────────────
ON_CODE_RE    = re.compile(r"^([A-Z]\d{3,5})\b")
ON_HASH_RE    = re.compile(r"^#\s+([A-Z]\d{3,5})\b")   # "# S207 Appendectomy..."
ON_SKIP_LINES = re.compile(r"^(Amd|February|March|April|May|June|July|August|"
                            r"September|October|November|December|NOT ALLOCATED)\b")
ON_PAGENO_RE  = re.compile(r"^[A-Z]{1,3}\d+\s*$")   # "GP8", "A1"
# Numeric/Alphabetic Index starts at page 829 — skip it entirely
ON_INDEX_START = 829

_ON_CHAPTER_MAP = [
    ("CONSULTATIONSANDVISITS",          "A – Consultations and Visits"),
    ("NUCLEARMEDICINEINVIVO",           "B – Nuclear Medicine In Vivo"),
    ("NUCLEARMEDICINEINVITRO",          "C – Nuclear Medicine In Vitro"),
    ("DIAGNOSTICRADIOLOGY",             "D – Diagnostic Radiology"),
    ("SPECIALPROCEDURES",               "E – Special Procedures"),
    ("PHYSICIANSANAESTHESIA",           "F – Anaesthesia"),
    ("GENERALSURGERY",                  "G – General Surgery"),
    ("DIGESTIVESYSTEM",                 "G – General Surgery"),   # surgical sub-section
    ("OBSTETRICS",                      "H – Obstetrics"),
    ("PATHOLOGY",                       "J – Pathology"),
    ("OPHTHALMOLOGY",                   "K – Ophthalmology"),
    ("PSYCHIATRY",                      "L – Psychiatry"),
    ("PAEDIATRICS",                     "M – Paediatrics"),
    ("ORTHOPEDICSURGERY",               "N – Orthopedic Surgery"),
    ("CARDIOTHORACICSURGERY",           "P – Cardiothoracic Surgery"),
    ("NEUROSURGERY",                    "Q – Neurosurgery"),
    ("UROLOGY",                         "R – Urology"),
    ("MISCELLANEOUS",                   "S – Miscellaneous"),
    ("PREMIUMPAYMENTS",                 "T – Premiums"),
    ("PHYSICIANPROGRAMSERVICES",        "U – Physician Program Services"),
    ("GENERALCONSIDERATIONS",           "GP – General Preamble"),
    ("GENERALPREAMBLE",                 "GP – General Preamble"),
    ("SURGICALPREAMBLE",                "G – General Surgery"),
    ("INTENSIVECORONARYCARE",           "IC – Intensive/Coronary Care"),
    ("RADIATIONONCOLOGY",               "X – Radiation Oncology"),
    ("ENDOCRINE",                       "G – General Surgery"),
    ("VASCULAR",                        "G – General Surgery"),
    ("THORACIC",                        "P – Cardiothoracic Surgery"),
    ("PLASTIC",                         "S – Miscellaneous"),
]

def _on_chapter(rh: str) -> str:
    t = rh.replace(" ", "").upper()
    for key, label in _ON_CHAPTER_MAP:
        if key in t:
            return label
    return rh if rh else ""


def _on_is_heading(text: str) -> bool:
    if ON_CODE_RE.match(text):
        return False
    if ON_HASH_RE.match(text):
        return False
    # Reject short tokens that are page refs (e.g. "I.C", "A1", "GP8")
    if len(text) <= 4:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.80 and len(text.split()) <= 18


def _on_make_record(code: str, title: str, price, chapter: str,
                    section: str, pn: int, x0: float) -> dict:
    return {
        "province":        "ON",
        "fsc_code":        code,
        "fsc_fn":          title,
        "fsc_chapter":     chapter,
        "fsc_section":     section,
        "fsc_subsection":  "",
        "fsc_description": "",
        "fsc_notes":       "",
        "fsc_others":      "",
        "price":           price,
        "page":            pn,
        "entry_x0":        x0,
    }


def parse_ontario(layout_path: Path) -> list:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    records = []
    current = None
    chapter = ""
    section = ""
    fn_pending = False   # code alone on its own line; next line → fsc_fn

    for page in data:
        pn = page["page"]

        # Skip the Numeric / Alphabetic Index at the back of the document
        if pn >= ON_INDEX_START:
            continue

        for line in page["lines"]:
            text = clean(line.get("text", ""))
            x0   = line.get("x0") or 0
            sz   = line.get("font_size") or 12

            if not text:
                continue

            # Skip metadata / footer lines
            if ON_SKIP_LINES.match(text):
                continue
            if re.match(r"^\d{4}\s*\(effective", text):
                continue

            # ── Format 1: "# CODE description"  (surgical/procedure schedule)
            #    x0 ~ 32, code preceded by hash + space
            mh = ON_HASH_RE.match(text) if x0 < 45 else None
            if mh:
                if current:
                    records.append(current)
                code  = mh.group(1)
                rest  = text[mh.end():].strip()
                # Remove trailing dot-leaders and extract price
                title, price = split_price(strip_dots(rest))
                current    = _on_make_record(code, title, price, chapter, section, pn, x0)
                fn_pending = not bool(title)
                continue

            # ── Format 2: plain "CODE [description]"  (consultation / visit schedule)
            m = ON_CODE_RE.match(text) if x0 < 60 else None
            if m:
                if current:
                    records.append(current)
                code  = m.group(1)
                rest  = text[m.end():].strip()
                title, price = split_price(rest)
                current    = _on_make_record(code, title, price, chapter, section, pn, x0)
                fn_pending = not bool(title)
                continue

            # Skip page-number-style tokens like "GP8", "A1", "S23"
            if ON_PAGENO_RE.match(text):
                continue

            # Running chapter heading — two layouts:
            #   Consultation schedule: centred at x0 > 130, sz 18
            #   Surgical schedule:    left-ish at x0 > 85, sz 18
            # Both are ALL-CAPS and NOT a fee-code line.
            t_nospace = text.replace(" ", "").upper()
            if x0 > 85 and sz >= 14 and _on_is_heading(text):
                new_ch = _on_chapter(t_nospace)
                if new_ch:
                    chapter = new_ch
                continue

            # Section heading at left margin (ALL-CAPS, no code)
            if x0 < 65 and _on_is_heading(text) and sz >= 10:
                if current:
                    records.append(current)
                    current = None
                fn_pending = False
                section = clean(text)
                continue

            # Price-only column (far right)
            if PRICE_ONLY_RE.match(text) and x0 > 350:
                if current and not current["price"]:
                    current["price"] = text.lstrip("$").strip()
                continue

            # CCP / column reference numbers (single digit at x0 > 400)
            # These are surgical schedule column values like "6", "7" for
            # anaesthesia units — not part of descriptions.
            if re.match(r"^\d{1,2}$", text) and x0 > 380:
                continue

            # ── Continuation / notes for current record ──────────────────────
            if current:
                t   = strip_dots(text)
                low = t.lower()

                # Classify line type
                is_payment_rule = bool(re.match(
                    r"(paymentrule|payment rule|paymentrules|payment rules)", low))
                is_commentary   = low.startswith("[commentary") or low.startswith("commentary")
                is_note_kw      = (is_payment_rule or is_commentary or
                                   low.startswith("note:") or
                                   bool(re.match(r"^\d+\.", low)) or
                                   bool(re.match(r"^[a-z]\.", low)) or
                                   low.startswith("claimssubmission") or
                                   low.startswith("claims submission"))

                if fn_pending and x0 >= 60 and not is_note_kw:
                    # First substantive continuation → fsc_fn
                    t_fn, t_price = split_price(t)
                    if not current["fsc_fn"]:
                        current["fsc_fn"] = t_fn
                        if t_price and not current["price"]:
                            current["price"] = t_price
                    else:
                        current["fsc_description"] = (
                            current["fsc_description"] + " " + t_fn).strip()
                    fn_pending = False

                elif is_note_kw:
                    current["fsc_notes"] = (current["fsc_notes"] + " " + t).strip()

                elif x0 >= 60:
                    # Indented continuation — description or commentary body
                    current["fsc_description"] = (
                        current["fsc_description"] + " " + t).strip()

                else:
                    # Left-margin continuation — wrapping title
                    if not current["fsc_fn"]:
                        current["fsc_fn"] = t
                    else:
                        current["fsc_description"] = (
                            current["fsc_description"] + " " + t).strip()

    if current:
        records.append(current)
    return records


# ── BRITISH COLUMBIA ──────────────────────────────────────────────────────────
BC_CODE_RE  = re.compile(r"^([A-Z]{0,2}\d{4,6})\b")
BC_SKIP_RE  = re.compile(r"^(Medical Services Commission|Anes\.|Level|\$)\b")
BC_PAGENO_RE = re.compile(r"^\d+-\d+$")

def _bc_is_section(sz: float, x0: float) -> bool:
    return sz > 10.4 and x0 < 82

def _bc_is_code(text: str, x0: float, sz: float) -> bool:
    return 60 < x0 < 82 and 9.5 < sz < 10.4 and bool(BC_CODE_RE.match(text))

def parse_bc(layout_path: Path) -> list:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    records = []
    current = None
    chapter   = ""
    section   = ""
    subsection = ""
    fn_pending = False   # True right after a code entry: next desc line → fsc_fn

    for page in data:
        pn = page["page"]
        for line in page["lines"]:
            text = clean(line.get("text", ""))
            x0   = line.get("x0") or 0
            sz   = line.get("font_size") or 10

            if not text:
                continue
            if BC_PAGENO_RE.match(text.strip()):
                continue
            if text.strip() in ("$", "Anes.", "Level", "Anes"):
                continue

            # Header line: "Medical Services Commission ... [Chapter Name]"
            if "Medical Services Commission" in text and x0 < 82:
                # extract chapter from right-side part
                parts = [p.strip() for p in text.split("\u00a0") if p.strip()]
                for kw in ["General Services", "Diagnostic", "Surgery", "Obstetrics",
                           "Anaesthesiology", "Ophthalmology", "Orthopaedics",
                           "Psychiatry", "Radiology", "Out-of-Office", "Cardiology",
                           "Neurology", "Urology", "Dermatology", "Paediatrics",
                           "Gynaecology", "Pathology", "Physical Medicine",
                           "Internal Medicine", "Ear", "Eyes"]:
                    if kw in text:
                        chapter = kw
                        break
                continue

            # Price column header
            if PRICE_ONLY_RE.match(text) and x0 > 400:
                if current and not current["price"]:
                    current["price"] = text.lstrip("$").strip()
                continue

            # Section heading (larger font, left-aligned)
            if _bc_is_section(sz, x0):
                if current:
                    records.append(current)
                    current = None
                fn_pending = False
                section = clean(text)
                subsection = ""
                continue

            # Sub-section markers like "(A) Acute renal failure"
            if 80 < x0 < 115 and sz >= 9.5:
                subsection = clean(text)
                continue

            # Fee code entry
            if _bc_is_code(text, x0, sz):
                if current:
                    records.append(current)
                m = BC_CODE_RE.match(text)
                code  = m.group(1)
                rest  = text[len(code):].strip()
                title, price = split_price(rest)
                current = {
                    "province": "BC",
                    "fsc_code": code,
                    "fsc_fn":   title,
                    "fsc_chapter":    chapter,
                    "fsc_section":    section,
                    "fsc_subsection": subsection,
                    "fsc_description": "",
                    "fsc_notes":  "",
                    "fsc_others": "",
                    "price":      price,
                    "page":       pn,
                    "entry_x0":   x0,
                }
                fn_pending = not bool(title)   # need next line for fn?
                continue

            # Description / notes for current record
            if current and x0 > 115:
                t = strip_dots(text)
                is_note_sz = sz < 9.7      # smaller font = note
                is_note_kw = t.lower().startswith("note")
                is_roman   = bool(re.match(r"^(i{1,4}v?|vi{0,3}|ix|x{0,3})\)", t, re.I))
                is_numitem = bool(re.match(r"^\d+\)", t))

                if fn_pending and not is_note_sz and not is_note_kw:
                    # First substantive line after code → becomes fsc_fn
                    # Also extract price if embedded (e.g. "Description   42.50")
                    t_fn, t_price = split_price(t)
                    if not current["fsc_fn"]:
                        current["fsc_fn"] = t_fn
                        if t_price and not current["price"]:
                            current["price"] = t_price
                    else:
                        current["fsc_description"] = (
                            current["fsc_description"] + " " + t_fn).strip()
                    fn_pending = False
                elif is_note_sz or is_note_kw or is_roman or is_numitem:
                    current["fsc_notes"] = (current["fsc_notes"] + " " + t).strip()
                else:
                    current["fsc_description"] = (
                        current["fsc_description"] + " " + t).strip()

    if current:
        records.append(current)
    return records


# ── YUKON ─────────────────────────────────────────────────────────────────────
YT_CODE_RE = re.compile(r"^(\d{4,6})\b")
YT_SKIP_RE = re.compile(r"^(These fees|YHCIP|YWCHSB|Anes|or Proc|Code|Page \d+)")

def _yt_is_chapter(sz: float) -> bool:
    return sz >= 14

def _yt_is_subcategory(text: str, x0: float, sz: float) -> bool:
    if x0 > 55 or sz > 12:
        return False
    t = text.strip()
    if YT_CODE_RE.match(t):
        return False
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio > 0.80 or t.endswith(":")


def parse_yukon(layout_path: Path) -> list:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    records = []
    current = None
    chapter   = ""
    section   = ""
    subsection = ""
    fn_pending = False

    for page in data:
        pn = page["page"]
        for line in page["lines"]:
            text = clean(line.get("text", ""))
            x0   = line.get("x0") or 0
            sz   = line.get("font_size") or 9

            if not text:
                continue
            if YT_SKIP_RE.match(text):
                continue

            # Price column value
            if PRICE_ONLY_RE.match(text) and x0 > 350:
                if current and not current["price"]:
                    current["price"] = text.lstrip("$").strip()
                continue

            # Chapter heading (large font)
            if _yt_is_chapter(sz):
                if current:
                    records.append(current)
                    current = None
                fn_pending = False
                chapter = clean(text)
                section = ""
                subsection = ""
                continue

            # Sub-category / section
            if _yt_is_subcategory(text, x0, sz):
                if current:
                    records.append(current)
                    current = None
                fn_pending = False
                t = clean(text).rstrip(":")
                # ALL CAPS → section, Title Case with colon → subsection
                letters = [c for c in t if c.isalpha()]
                ur = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
                if ur > 0.80:
                    section    = t
                    subsection = ""
                else:
                    subsection = t
                continue

            # Fee code entry
            m = YT_CODE_RE.match(text)
            if m and 35 <= x0 <= 80 and sz < 11:
                if current:
                    records.append(current)
                code  = m.group(1)
                rest  = text[len(code):].strip()
                title, price = split_price(rest)
                current = {
                    "province":  "YT",
                    "fsc_code":  code,
                    "fsc_fn":    title,
                    "fsc_chapter":    chapter,
                    "fsc_section":    section,
                    "fsc_subsection": subsection,
                    "fsc_description": "",
                    "fsc_notes":  "",
                    "fsc_others": "",
                    "price":      price,
                    "page":       pn,
                    "entry_x0":   x0,
                }
                fn_pending = not bool(title)
                continue

            # Description / notes
            if current and x0 >= 90:
                t = strip_dots(text)
                low = t.lower()
                is_note = (low.startswith("note") or re.match(r"^\d+\.", low)
                           or re.match(r"^[a-z]\.", low))
                if fn_pending and not is_note:
                    if not current["fsc_fn"]:
                        current["fsc_fn"] = t
                    else:
                        current["fsc_description"] = (
                            current["fsc_description"] + " " + t).strip()
                    fn_pending = False
                elif is_note:
                    current["fsc_notes"] = (current["fsc_notes"] + " " + t).strip()
                else:
                    current["fsc_description"] = (
                        current["fsc_description"] + " " + t).strip()

    if current:
        records.append(current)
    return records


# ── post-processing ──────────────────────────────────────────────────────────

_ON_SECTION_CLEAN = re.compile(r"([a-z])([A-Z])")   # "paymentRules" → "payment Rules"

def _clean_on_section(s: str) -> str:
    """Insert space before capital letters in run-together ALL-CAPS heading."""
    # e.g. "DETENTIONINAMBULANCE" → keep as-is (it's a heading label, not prose)
    return s.strip()


_NOISY_CHAPTERS = {
    "P": "P – Surgery",
    "IC": "IC – Intensive/Coronary Care",
    "I.C": "IC – Intensive/Coronary Care",
    "NOTALLOCATED": "Not Allocated",
    "APPENDIXF": "Appendix F",
    "WSIB": "WSIB",
    "RADIATIONONCOLOGY": "Radiation Oncology",
    "ENDOCRINESURGICALPROCEDURES": "Endocrine Surgical Procedures",
    "GENERALPREAMBLE": "GP – General Preamble",
}

def post_process(records: list) -> list:
    """Clean up chapter names and prices embedded in fsc_fn."""
    PRICE_TRAIL = re.compile(r"\s+\$?(\d{1,6}\.\d{2})\s*$")
    for r in records:
        # Fix embedded price in fsc_fn (happens when price is at end of desc line)
        m = PRICE_TRAIL.search(r["fsc_fn"])
        if m:
            r["price"] = r["price"] or m.group(1)
            r["fsc_fn"] = r["fsc_fn"][: m.start()].strip(" .")

        # Normalise chapter
        ch = r["fsc_chapter"].strip()
        r["fsc_chapter"] = _NOISY_CHAPTERS.get(ch, ch)

        # Remove stray page-number leakage like ",GP54" or "GP19)."
        if re.match(r"^,?GP\d", r["fsc_chapter"]):
            r["fsc_chapter"] = "GP – General Preamble"
        if re.match(r"^GP\d+", r["fsc_chapter"]):
            r["fsc_chapter"] = "GP – General Preamble"

        # Trim trailing whitespace in all string fields
        for f in ("fsc_fn", "fsc_description", "fsc_notes", "fsc_others",
                  "fsc_section", "fsc_subsection", "fsc_chapter"):
            r[f] = r[f].strip()
    return records


# ── runner ────────────────────────────────────────────────────────────────────

def deduplicate(records: list) -> list:
    seen = {}
    out  = []
    for r in records:
        key = (r["province"], r["fsc_code"])
        if key not in seen:
            seen[key] = len(out)
            out.append(r)
        else:
            # merge description/notes into existing record if it's richer
            existing = out[seen[key]]
            for field in ("fsc_fn", "fsc_description", "fsc_notes"):
                if not existing[field] and r[field]:
                    existing[field] = r[field]
            if not existing["price"] and r["price"]:
                existing["price"] = r["price"]
    return out


def main():
    parsers = {
        "ON": (LAYOUT_DIR / "on_layout.json",  parse_ontario),
        "BC": (LAYOUT_DIR / "bc_layout.json",  parse_bc),
        "YT": (LAYOUT_DIR / "yt_layout.json",  parse_yukon),
    }

    all_records = []
    for province, (layout_path, parser_fn) in parsers.items():
        if not layout_path.exists():
            print(f"[SKIP] {layout_path.name} not found")
            continue
        print(f"[{province}] Parsing...", flush=True)
        records   = parser_fn(layout_path)
        records   = post_process(records)
        deduped   = deduplicate(records)
        no_fn     = sum(1 for r in deduped if not r["fsc_fn"])
        no_price  = sum(1 for r in deduped if not r["price"])
        print(f"[{province}] {len(records)} raw -> {len(deduped)} unique "
              f"| missing fn={no_fn} price={no_price}")
        out_path = OUT_DIR / f"{province.lower()}_fee_codes.json"
        out_path.write_text(
            json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        all_records.extend(deduped)

    combined = OUT_DIR / "all_fee_codes.json"
    combined.write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTotal: {len(all_records)} fee codes -> {combined}")


if __name__ == "__main__":
    main()
