"""
Map every parsed FSC code to an NGS category.

Strategy (applied in order, first match wins):
  1. Role-code rule  – if code ends in 'B' (ON) or role-code table → 073/075
  2. Keyword rules   – pattern matching on description + section
  3. Text similarity – Jaccard + overlap on description vs NGS label+description
  4. Section rules   – map entire section to a broad NGS bucket if no better match

Outputs:
  data/parsed/fsc_ngs_mapped.json   – all fee codes with NGS mapping fields added
"""
import json
import re
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
PARSED   = BASE_DIR / "data" / "parsed"
NGS_DIR  = PARSED / "ngs"
FC_DIR   = PARSED / "fee_codes"


# ── load reference data ───────────────────────────────────────────────────────

def load_ngs() -> list[dict]:
    return json.loads((NGS_DIR / "ngs_categories.json").read_text(encoding="utf-8"))

def load_ccp_map() -> list[dict]:
    return json.loads((NGS_DIR / "ccp_ngs_map.json").read_text(encoding="utf-8"))

def load_role_tables() -> list[dict]:
    return json.loads((NGS_DIR / "role_code_tables.json").read_text(encoding="utf-8"))

def load_fee_codes() -> list[dict]:
    return json.loads((FC_DIR / "all_fee_codes.json").read_text(encoding="utf-8"))


# ── text helpers ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","and","or","of","for","to","in","on","with","by","at",
    "from","as","is","are","be","this","that","per","each","visit","service",
    "services","procedure","procedures","full","quarter","hour","payment",
    "rules","including","only","when","not","no","any","all","if","other",
    "than","also","where","may","used","use","within","after","before","without",
    "during","which","who","same","more","one","two","three","four","five",
}

def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]

def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def overlap(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), 1)

def text_score(fee_text: str, ngs_text: str) -> tuple[float, list[str]]:
    ft = tokenize(fee_text)
    nt = tokenize(ngs_text)
    shared = sorted(set(ft) & set(nt))
    score  = 0.35 * jaccard(ft, nt) + 0.65 * overlap(ft, nt)
    return score, shared

def confidence_label(score: float) -> str:
    if score >= 0.55:   return "high"
    if score >= 0.30:   return "medium"
    if score >= 0.15:   return "low"
    return "very_low"


# ── keyword rules ─────────────────────────────────────────────────────────────
# Each rule: (regex pattern on combined fee text, ngs_code, rationale)
# Ordered from most specific to most general.

_KW_RULES = [
    # Surgical assistance / anesthesia (global rules)
    (r"\bsurgical assist",          "073.000", "Surgical assistance service"),
    (r"\bassistant.*surg",          "073.000", "Surgical assistance service"),
    (r"\banaesth|anesthesia\b",     "075.000", "Anaesthesia/anesthesia service"),

    # Consultations
    (r"\bmajor consult",            "001.000", "Major consultation"),
    (r"\bsurgical consult",         "001.000", "Surgical consultation"),
    (r"\bconsult",                  "002.000", "Consultation service"),

    # Assessments
    (r"\bgeneral assessment|general re.?assess",
                                    "003.000", "General/specific assessment"),
    (r"\bspecific assessment",      "003.000", "Specific assessment"),
    (r"\bannual.*exam|annual health exam",
                                    "003.000", "Annual health exam"),
    (r"\bspecial eye|oculo.?visual|refraction",
                                    "009.000", "Special eye assessment"),
    (r"\bminor assess|partial assess|limited consult|repeat consult",
                                    "010.000", "Other/minor assessment"),
    (r"\bsubsequent.*visit|follow.?up",
                                    "010.000", "Subsequent office visit"),
    (r"\bhospital.*inpatient.*assess|inpatient.*assess",
                                    "005.000", "Hospital inpatient assessment"),
    (r"\bnewborn|premature.*baby|low birth weight",
                                    "004.000", "Newborn/premature care"),
    (r"\bhospital.*outpatient|outpatient.*visit",
                                    "006.000", "Hospital outpatient"),
    (r"\bhome.*visit|patient.*home",
                                    "021.000", "Home visit"),

    # Special calls / detention
    (r"\bspecial call|out.?of.?hours|out of hours|after.?hours|evening.*premium|night.*premium",
                                    "020.000", "Out-of-hours/special call"),
    (r"\bdetention",                "016.000", "Detention"),
    (r"\bpalliative",               "017.000", "Palliative care"),

    # Psychiatric / counselling
    (r"\bindividual psycho|psychiatry|psychiatric",
                                    "022.000", "Individual psychiatry"),
    (r"\bgroup.*psycho|family.*psycho",
                                    "023.000", "Group/family psychiatry"),
    (r"\bcounsell?ing",             "024.000", "Counselling"),

    # Surgery – major categories
    (r"\bmastectomy|breast.*surg|breast.*excision",
                                    "025.000", "Mastectomy/breast surgery"),
    (r"\bbiopsy.*breast|breast.*biopsy|breast.*tumour|breast.*tumor",
                                    "026.000", "Breast tumor/biopsy"),
    (r"\bfracture|dislocation",     "028.000", "Fracture/dislocation"),
    (r"\bdisc.*surg|vertebr|spinal.*canal|laminect",
                                    "029.000", "Disc/vertebrae surgery"),
    (r"\bhip.*arthroplasty|hip.*replac",
                                    "030.000", "Hip arthroplasty"),
    (r"\bknee.*arthroplasty|knee.*replac",
                                    "031.000", "Knee arthroplasty"),
    (r"\bcoronary.*bypass|cabg|bypass graft",
                                    "036.000", "Coronary artery bypass"),
    (r"\bangioplasty|percutaneous.*coronary",
                                    "037.000", "Coronary angioplasty"),
    (r"\belectrophysio|pacemaker|defibrillat|ablation",
                                    "038.000", "Cardiac electrophysiology"),
    (r"\bappendectomy|appendix",    "043.000", "Appendectomy"),
    (r"\blaparotomy",               "044.000", "Laparotomy"),
    (r"\bgallbladder|cholecyst|biliary",
                                    "045.000", "Gallbladder/biliary"),
    (r"\btonsil|adenoid",           "046.000", "Tonsillectomy"),
    (r"\bhernia",                   "047.000", "Hernia"),
    (r"\bcolectomy|colon.*resect|intestin",
                                    "048.000", "Colon/intestines"),
    (r"\bhemorrhoid|proctotomy|rectum|anus",
                                    "049.000", "Rectum/hemorrhoidectomy"),
    (r"\bprostatectomy|prostate.*surg",
                                    "052.000", "Prostate surgery"),
    (r"\bvasectomy",                "053.000", "Vasectomy"),
    (r"\bhysterectomy",             "056.000", "Hysterectomy"),
    (r"\bsteriliz|tubal",           "057.000", "Sterilization (female)"),
    (r"\bcaesarean|c.?section|obstetric",
                                    "059.000", "Caesarean section"),
    (r"\bcataract",                 "065.000", "Cataract surgery"),
    (r"\btonsillectomy",            "046.000", "Tonsillectomy"),
    (r"\bnephrectomy|kidney.*surg",  "051.000", "Urinary system surgery"),

    # Diagnostic / lab
    (r"\bx.?ray|radiograph|fluoroscop",
                                    "080.000", "X-ray/radiography"),
    (r"\bct scan|computed tomograph",
                                    "081.000", "CT scan"),
    (r"\bmri|magnetic resonance",   "082.000", "MRI"),
    (r"\bultrasound|echograph",     "083.000", "Ultrasound"),
    (r"\bnuclear.*med|scintigraph",  "084.000", "Nuclear medicine imaging"),
    (r"\becg|electrocardiogram|electrocardiograph",
                                    "088.000", "ECG"),
    (r"\blab.*test|laboratory|blood test|urinalysis",
                                    "096.000", "Laboratory tests"),

    # Rehabilitation / physical medicine
    (r"\bphysio|physical.*ther|rehabilit",
                                    "077.000", "Rehabilitation/physiotherapy"),

    # Premiums / add-ons (broad catch)
    (r"\bpremium|surcharge|add.?on|\badd\b",
                                    "015.000", "Premium/add-on"),
]

_COMPILED_RULES = [
    (re.compile(pat, re.IGNORECASE), ngs, rat)
    for pat, ngs, rat in _KW_RULES
]


# ── section → NGS fallback ────────────────────────────────────────────────────

_SECTION_FALLBACK = {
    # ON section patterns → broad NGS
    "DETENTION":            "016.000",
    "CONSULTATION":         "002.000",
    "ASSESSMENT":           "003.000",
    "ANAESTHESIA":          "075.000",
    "OBSTETRIC":            "059.000",
    "PAEDIATRIC":           "003.000",
    "PSYCHIATRY":           "022.000",
    "RADIOLOGY":            "080.000",
    "PATHOLOGY":            "096.000",
    "SURGERY":              "032.000",
    "OPHTHALMOLOGY":        "065.000",
    "CARDIOLOGY":           "039.000",
    "DERMATOLOGY":          "027.000",
    "NEUROLOGY":            "032.000",
    # BC chapters
    "Out-of-Office":        "020.000",
    "General Services":     "010.000",
    "Diagnostic":           "080.000",
    "Anaesthesiology":      "075.000",
    "Obstetrics":           "059.000",
    "Psychiatry":           "022.000",
    "Radiology":            "080.000",
    "Ophthalmology":        "065.000",
    "Orthopaedics":         "032.000",
    "Dermatology":          "027.000",
    "Cardiology":           "039.000",
    # YT chapters
    "Communication":        "010.000",
}


def _section_fallback_ngs(section: str, chapter: str) -> Optional[str]:
    combined = (section + " " + chapter).upper()
    for kw, ngs in _SECTION_FALLBACK.items():
        if kw.upper() in combined:
            return ngs
    return None


# ── per-province FSC → fee code decoding ────────────────────────────────────

def decode_fsc(fsc_code: str, province: str) -> tuple[str, str]:
    """Return (fee_code, role_code) from an FSC code string."""
    fsc = fsc_code.strip()
    if province == "ON":
        # First 4 chars = fee code, char 5 = role code
        if len(fsc) >= 5:
            return fsc[:4], fsc[4]
        return fsc, "A"
    elif province == "BC":
        # BC FSC: 7 chars; drop first 2 leading zeros → 5-char fee code
        # But in our parsed data the code already is the fee code (from PDF)
        # Strip leading zeros from numeric part
        num = re.sub(r"^0{1,2}", "", fsc)
        return num if num else fsc, ""
    elif province == "YT":
        # 4-digit or 6-digit starting 43/70
        return fsc, ""
    return fsc, ""


# ── role code to NGS ──────────────────────────────────────────────────────────

_ROLE_NGS = {}   # (province, role_code) → ngs_code (populated from role tables)

def build_role_ngs_index(role_tables: list):
    for r in role_tables:
        prov = r["province"]
        rc   = r["role_code"].strip()
        ngs  = r["ngs_codes"]
        if ngs:
            _ROLE_NGS[(prov, rc)] = ngs[0]
            _ROLE_NGS[("ALL", rc)] = ngs[0]   # fallback


# ── main mapping function ─────────────────────────────────────────────────────

def map_fee_to_ngs(
    fee: dict,
    ngs_list: list[dict],
    top_n: int = 3,
) -> dict:
    """Return the best NGS match dict for a fee code record."""
    province = fee["province"]
    fsc_code = fee["fsc_code"]
    _, role_code = decode_fsc(fsc_code, province)

    # Combined text for matching
    combined_text = " ".join([
        fee.get("fsc_fn", ""),
        fee.get("fsc_description", ""),
        fee.get("fsc_section", ""),
        fee.get("fsc_chapter", ""),
    ])

    result = {
        "NGS_code":             "",
        "NGS_label":            "",
        "NGS_rationale":        "",
        "NGS_confidence":       "very_low",
        "NGS_key_observations": "",
        "NGS_notes":            "",
        "NGS_other":            "",
        "match_method":         "",
        "match_score":          0.0,
    }

    # ── Rule 1: role-code lookup ──
    if role_code:
        ngs_from_role = (_ROLE_NGS.get((province, role_code))
                         or _ROLE_NGS.get(("ALL", role_code)))
        if ngs_from_role:
            ngs_entry = next((n for n in ngs_list
                              if n["ngs_code"] == ngs_from_role), None)
            result.update({
                "NGS_code":       ngs_from_role,
                "NGS_label":      ngs_entry["ngs_label"] if ngs_entry else "",
                "NGS_rationale":  f"Role code {role_code!r} maps to NGS {ngs_from_role}",
                "NGS_confidence": "high",
                "match_method":   "role_code_table",
                "match_score":    1.0,
            })
            return result

    # ── Rule 2: keyword rules ──
    for pattern, ngs_code, rationale in _COMPILED_RULES:
        if pattern.search(combined_text):
            ngs_entry = next((n for n in ngs_list
                              if n["ngs_code"] == ngs_code), None)
            result.update({
                "NGS_code":       ngs_code,
                "NGS_label":      ngs_entry["ngs_label"] if ngs_entry else "",
                "NGS_rationale":  rationale,
                "NGS_confidence": "high",
                "match_method":   "keyword_rule",
                "match_score":    0.9,
            })
            return result

    # ── Rule 3: text similarity against all NGS labels+descriptions ──
    best_score, best_ngs, best_shared = 0.0, None, []
    for ngs in ngs_list:
        ngs_text  = f"{ngs['ngs_label']} {ngs.get('ngs_description','')}"
        sc, shared = text_score(combined_text, ngs_text)
        if sc > best_score:
            best_score, best_ngs, best_shared = sc, ngs, shared

    if best_ngs and best_score >= 0.10:
        result.update({
            "NGS_code":             best_ngs["ngs_code"],
            "NGS_label":            best_ngs["ngs_label"],
            "NGS_rationale":        (f"Shared terms: {', '.join(best_shared[:8])}"
                                     if best_shared else "Lexical similarity"),
            "NGS_confidence":       confidence_label(best_score),
            "NGS_key_observations": f"Score={best_score:.3f}",
            "match_method":         "text_similarity",
            "match_score":          round(best_score, 4),
        })
        return result

    # ── Rule 4: section/chapter fallback ──
    fallback = _section_fallback_ngs(
        fee.get("fsc_section", ""), fee.get("fsc_chapter", "")
    )
    if fallback:
        ngs_entry = next((n for n in ngs_list if n["ngs_code"] == fallback), None)
        result.update({
            "NGS_code":       fallback,
            "NGS_label":      ngs_entry["ngs_label"] if ngs_entry else "",
            "NGS_rationale":  "Section/chapter keyword fallback",
            "NGS_confidence": "low",
            "match_method":   "section_fallback",
            "match_score":    0.2,
        })
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading reference data...", flush=True)
    ngs_list    = load_ngs()
    role_tables = load_role_tables()
    build_role_ngs_index(role_tables)

    print(f"  {len(ngs_list)} NGS categories, {len(role_tables)} role table entries")

    print("Loading fee codes...", flush=True)
    fee_codes = load_fee_codes()
    print(f"  {len(fee_codes)} total fee codes")

    print("Mapping FSC -> NGS...", flush=True)
    results = []
    method_counts: dict[str, int] = {}

    for fee in fee_codes:
        mapping = map_fee_to_ngs(fee, ngs_list)
        merged  = {**fee, **mapping}
        results.append(merged)
        m = mapping["match_method"] or "none"
        method_counts[m] = method_counts.get(m, 0) + 1

    out_path = PARSED / "fsc_ngs_mapped.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total   = len(results)
    no_ngs  = sum(1 for r in results if not r["NGS_code"])
    high    = sum(1 for r in results if r["NGS_confidence"] == "high")
    med     = sum(1 for r in results if r["NGS_confidence"] == "medium")
    low     = sum(1 for r in results if r["NGS_confidence"] in ("low","very_low"))

    print(f"\nMapped {total} codes:")
    print(f"  No NGS assigned : {no_ngs}")
    print(f"  High confidence : {high}")
    print(f"  Medium          : {med}")
    print(f"  Low / very low  : {low}")
    print(f"  Method breakdown: {method_counts}")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
