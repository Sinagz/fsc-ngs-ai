"""
Cross-province FSC similarity matching (v2 – strict bipartite matching).

Rules:
  - Each group contains AT MOST ONE code per province.
  - Two codes can be grouped only if from DIFFERENT provinces.
  - Similarity threshold: >= 0.55 for same NGS; >= 0.75 for different NGS.
  - Codes with empty description are only matched if BOTH are empty (same NGS bucket).
  - top_key format: "{ngs_code}_{seq:04d}"

Matching procedure (per NGS bucket):
  1. Build all cross-province pairs (ON-BC, ON-YT, BC-YT) with scores.
  2. Greedy: sort pairs by score desc; assign group if both sides still unassigned.
  3. Remaining unmatched codes get singleton top_keys.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import date

BASE_DIR = Path(__file__).resolve().parent.parent
PARSED   = BASE_DIR / "data" / "parsed"

PROVINCES = ["ON", "BC", "YT"]

STOPWORDS = {
    "the","a","an","and","or","of","for","to","in","on","with","by","at",
    "from","as","is","are","be","this","that","per","each","full","quarter",
    "hour","payment","rules","including","only","when","not","no","any","all",
    "if","other","than","also","where","may","used","use","within","after",
    "before","without","during","which","who","same","more","one","two",
    "three","four","five","six","seven","eight","nine","ten",
}

def tokenize(text: str) -> list:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]

def sim(ta: list, tb: list) -> float:
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    j = len(sa & sb) / len(sa | sb)
    o = len(sa & sb) / max(len(sa), 1)
    return 0.4 * j + 0.6 * o

def desc_tokens(r: dict) -> list:
    return tokenize(" ".join([
        r.get("fsc_fn", ""),
        r.get("fsc_description", ""),
    ]))


def add_fsc_meta(records: list) -> list:
    for r in records:
        fn    = r.get("fsc_fn", "")
        price = r.get("price")
        desc  = r.get("fsc_description", "")
        ch    = r.get("fsc_chapter", "")
        sec   = r.get("fsc_section", "")
        sub   = r.get("fsc_subsection", "")

        if fn and price:        conf = "high"
        elif fn:                conf = "medium"
        elif desc:              conf = "low"
        else:                   conf = "very_low"

        parts = []
        if ch:    parts.append(f"chapter: {ch[:35]}")
        if sec:   parts.append(f"section: {sec[:30]}")
        if sub:   parts.append(f"subsection: {sub[:25]}")
        if price: parts.append(f"price: {price}")

        r["fsc_rationale"]        = "; ".join(parts)
        r["fsc_confidence"]       = conf
        r["fsc_key_observations"] = (
            f"Page {r.get('page','')} | x0={r.get('entry_x0','')}"
        )
    return records


def build_top_keys(records: list) -> list:
    """Assign top_key with strict 1-per-province grouping."""

    # Pre-compute tokens
    tok = [desc_tokens(r) for r in records]
    n   = len(records)

    # Group records by NGS code
    ngs_buckets: dict[str, list] = defaultdict(list)
    for i, r in enumerate(records):
        ngs = r.get("NGS_code") or "NOMAP"
        ngs_buckets[ngs].append(i)

    top_key: list = [None] * n    # top_key[i] = assigned key string

    global_seq: dict[str, int] = {}   # ngs_code → next sequence number

    def next_seq(ngs: str) -> str:
        s = global_seq.get(ngs, 0) + 1
        global_seq[ngs] = s
        return f"{ngs}_{s:04d}"

    print("  Running bipartite matching per NGS bucket...", flush=True)
    total_groups = 0
    total_cross  = 0

    for ngs, indices in ngs_buckets.items():
        if ngs == "NOMAP" or len(indices) < 2:
            continue

        # Separate by province
        by_prov: dict[str, list] = defaultdict(list)
        for i in indices:
            by_prov[records[i]["province"]].append(i)

        # Build candidate pairs across provinces
        pairs = []   # (score, i, j)
        prov_list = list(by_prov.keys())
        for pi in range(len(prov_list)):
            for pj in range(pi + 1, len(prov_list)):
                pa, pb = prov_list[pi], prov_list[pj]
                for i in by_prov[pa]:
                    for j in by_prov[pb]:
                        ta, tb = tok[i], tok[j]
                        # Both empty: low-confidence NGS-only match
                        if not ta and not tb:
                            sc = 0.05
                        elif not ta or not tb:
                            sc = 0.02   # one-sided empty → skip
                        else:
                            sc = sim(ta, tb)
                        if sc >= 0.02:
                            pairs.append((sc, i, j))

        # Sort descending by score
        pairs.sort(key=lambda x: -x[0])

        # Greedy assignment: each index can be in at most one group
        # and each group has at most 1 per province
        assigned: dict = {}   # index → group_key
        group_provs: dict = defaultdict(set)   # group_key → set of provinces used

        for sc, i, j in pairs:
            # Determine threshold
            threshold = 0.55 if (tok[i] and tok[j]) else 0.03

            if sc < threshold:
                continue

            ki = assigned.get(i)
            kj = assigned.get(j)
            pi = records[i]["province"]
            pj = records[j]["province"]

            if ki is None and kj is None:
                # Both unassigned → new group
                key = next_seq(ngs)
                assigned[i] = key
                assigned[j] = key
                group_provs[key].add(pi)
                group_provs[key].add(pj)
                total_cross += 1

            elif ki is not None and kj is None:
                # i already in a group; add j if province not yet in that group
                if pj not in group_provs[ki]:
                    assigned[j] = ki
                    group_provs[ki].add(pj)

            elif ki is None and kj is not None:
                if pi not in group_provs[kj]:
                    assigned[i] = kj
                    group_provs[kj].add(pi)

            # Both already assigned → skip (don't merge groups)

        # Apply assignments
        for i in indices:
            if assigned.get(i):
                top_key[i] = assigned[i]
                total_groups += 1

    # Assign singleton keys to unmatched records
    for i, r in enumerate(records):
        if top_key[i] is None:
            ngs = r.get("NGS_code") or "NOMAP"
            if ngs == "NOMAP":
                top_key[i] = f"NOMAP_{r['province']}_{r['fsc_code']}"
            else:
                top_key[i] = next_seq(ngs)

    for i, r in enumerate(records):
        r["top_key"] = top_key[i]

    from collections import Counter
    kc = Counter(top_key)
    cross_groups = sum(1 for v in kc.values() if v > 1)
    singletons   = sum(1 for v in kc.values() if v == 1)
    max_size     = max(kc.values())
    print(f"  {len(kc)} top_keys: {cross_groups} cross-province groups, "
          f"{singletons} singletons, max_group={max_size}")

    return records


def main():
    in_path  = PARSED / "fsc_ngs_mapped.json"
    out_path = PARSED / "fsc_ngs_cross_province.json"

    print("Loading mapped fee codes...", flush=True)
    records = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"  {len(records)} records")

    print("Adding FSC metadata...", flush=True)
    records = add_fsc_meta(records)

    print("Running cross-province matching...", flush=True)
    records = build_top_keys(records)

    out_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Saved -> {out_path}")

    # Show some examples of good cross-province matches
    from collections import Counter
    kc = Counter(r["top_key"] for r in records)
    cross_keys = [k for k, v in kc.items() if v > 1 and v <= 4]
    print(f"\nSample cross-province groups (size 2-4, showing first 8):")
    shown = 0
    for key in cross_keys:
        members = [r for r in records if r["top_key"] == key]
        if len(set(r["province"] for r in members)) < 2:
            continue
        print(f"\n  top_key={key}")
        for m in members:
            print(f"    [{m['province']}] {m['fsc_code']:10} | {m['fsc_fn'][:50]} | NGS={m['NGS_code']}")
        shown += 1
        if shown >= 8:
            break


if __name__ == "__main__":
    main()
