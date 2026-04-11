"""
Core lookup engine — semantic search with BGE-large embeddings.

Primary API:
    result = engine.search(fsc_code, input_province, output_province, top_n)

Falls back to Jaccard+overlap similarity if embeddings are not yet built.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "parsed" / "fsc_ngs_mapped.json"
NGS_PATH    = BASE_DIR / "data" / "parsed" / "ngs" / "ngs_categories.json"
EMBED_PATH  = BASE_DIR / "data" / "parsed" / "embeddings.npz"

PROVINCES   = ["ON", "BC", "YT"]

# ── text helpers (used as fallback when embeddings not available) ─────────────

STOPWORDS = {
    "the","a","an","and","or","of","for","to","in","on","with","by","at",
    "from","as","is","are","be","this","that","per","each","full","quarter",
    "hour","payment","rules","including","only","when","not","no","any","all",
    "if","other","than","also","where","may","used","use","within","after",
    "before","without","during","which","who","same","more","one","two",
    "three","four","five","six","seven","eight","nine","ten",
}

def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]

def _jaccard_overlap(ta: list[str], tb: list[str]) -> float:
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    j = len(sa & sb) / len(sa | sb)
    o = len(sa & sb) / max(len(sa), 1)
    return round(0.4 * j + 0.6 * o, 4)

def _rec_text(r: dict) -> str:
    parts = [
        r.get("fsc_fn", ""),
        r.get("fsc_description", ""),
        r.get("fsc_section", ""),
        r.get("fsc_subsection", ""),
        r.get("fsc_chapter", ""),
        r.get("NGS_label", ""),
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class FeeCode:
    province: str
    fsc_code: str
    fsc_fn: str
    fsc_chapter: str
    fsc_section: str
    fsc_subsection: str
    fsc_description: str
    fsc_notes: str
    fsc_others: str
    price: str
    page: str | int
    fsc_rationale: str
    fsc_confidence: str
    fsc_key_observations: str
    NGS_code: str
    NGS_label: str
    NGS_rationale: str
    NGS_confidence: str
    NGS_key_observations: str
    NGS_notes: str
    NGS_other: str
    match_method: str
    _raw: dict = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "FeeCode":
        return cls(
            province             = d.get("province", ""),
            fsc_code             = d.get("fsc_code", ""),
            fsc_fn               = d.get("fsc_fn", ""),
            fsc_chapter          = d.get("fsc_chapter", ""),
            fsc_section          = d.get("fsc_section", ""),
            fsc_subsection       = d.get("fsc_subsection", ""),
            fsc_description      = d.get("fsc_description", ""),
            fsc_notes            = d.get("fsc_notes", ""),
            fsc_others           = d.get("fsc_others", ""),
            price                = str(d.get("price") or ""),
            page                 = d.get("page", ""),
            fsc_rationale        = d.get("fsc_rationale", ""),
            fsc_confidence       = d.get("fsc_confidence", ""),
            fsc_key_observations = d.get("fsc_key_observations", ""),
            NGS_code             = d.get("NGS_code", ""),
            NGS_label            = d.get("NGS_label", ""),
            NGS_rationale        = d.get("NGS_rationale", ""),
            NGS_confidence       = str(d.get("NGS_confidence") or ""),
            NGS_key_observations = d.get("NGS_key_observations", ""),
            NGS_notes            = d.get("NGS_notes", ""),
            NGS_other            = str(d.get("NGS_other") or ""),
            match_method         = d.get("match_method", ""),
            _raw                 = d,
        )


@dataclass
class MatchResult:
    fee_code: FeeCode
    sim_score: float        # 0.0 – 1.0
    ngs_match: bool         # same NGS code as anchor
    score_method: str       # "semantic" | "jaccard"
    # Note: ngs_match=False is NOT an error — two equivalent codes in different
    # provinces can legitimately belong to different NGS categories.


@dataclass
class LookupResult:
    anchor: FeeCode
    output_province: str
    matches: list[MatchResult]      # ranked list, output province only
    ngs_label: str
    ngs_description: str
    score_method: str               # "semantic" | "jaccard"


# ── engine ────────────────────────────────────────────────────────────────────

class LookupEngine:
    def __init__(self):
        self._records: list[dict]          = []
        self._ngs_map: dict[str, dict]     = {}
        self._by_prov: dict[str, list[int]] = {}   # province -> [record indices]
        self._embed: np.ndarray | None     = None  # (N, D) float32 L2-normalised
        self._embed_idx: np.ndarray | None = None  # (N,) int — maps embed row -> record idx
        self._loaded   = False
        self._has_embed = False

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, force_reload: bool = False):
        if self._loaded and not force_reload:
            return
        self._records = json.loads(DATA_PATH.read_text(encoding="utf-8"))

        for i, r in enumerate(self._records):
            self._by_prov.setdefault(r["province"], []).append(i)

        if NGS_PATH.exists():
            cats = json.loads(NGS_PATH.read_text(encoding="utf-8"))
            self._ngs_map = {c["ngs_code"]: c for c in cats}

        if EMBED_PATH.exists():
            npz = np.load(str(EMBED_PATH), allow_pickle=False)
            self._embed     = npz["embeddings"].astype(np.float32)
            self._embed_idx = npz["record_ids"].astype(np.int32)
            self._has_embed = True
        else:
            self._has_embed = False

        self._loaded = True

    @property
    def embedding_available(self) -> bool:
        return self._has_embed

    # ── public helpers ────────────────────────────────────────────────────────

    def provinces(self) -> list[str]:
        self.load()
        return [p for p in PROVINCES if p in self._by_prov]

    def all_codes_for_province(self, province: str) -> list[str]:
        self.load()
        return sorted({self._records[i]["fsc_code"]
                       for i in self._by_prov.get(province, [])})

    def fuzzy_search(self, query: str, province: str | None = None,
                     limit: int = 20) -> list[dict]:
        """Return records matching code fragment or description text."""
        self.load()
        q = query.strip().upper()
        if not q:
            return []
        results = []
        for r in self._records:
            if province and r["province"] != province:
                continue
            if q in r["fsc_code"].upper() or q.lower() in (r.get("fsc_fn") or "").lower():
                results.append(r)
            if len(results) >= limit:
                break
        return results

    # ── main search ───────────────────────────────────────────────────────────

    def search(
        self,
        fsc_code: str,
        input_province: str,
        output_province: str,
        top_n: int = 5,
    ) -> LookupResult | None:
        self.load()
        fsc_code = fsc_code.strip().upper()

        # Find anchor record
        anchor_idx  = None
        anchor_raw  = None
        for i in self._by_prov.get(input_province, []):
            if self._records[i]["fsc_code"].upper() == fsc_code:
                anchor_idx = i
                anchor_raw = self._records[i]
                break
        if anchor_raw is None:
            return None

        anchor     = FeeCode.from_dict(anchor_raw)
        anchor_ngs = anchor.NGS_code

        ngs_info    = self._ngs_map.get(anchor_ngs, {})
        ngs_label   = ngs_info.get("ngs_label",       anchor.NGS_label or "")
        ngs_desc    = ngs_info.get("ngs_description", "")

        # Score candidates in output province
        # We do NOT force same-NGS — two provinces can legitimately map equivalent
        # procedures to different NGS codes. We rank purely by description similarity
        # and flag same/different NGS as informational only.
        if self._has_embed:
            matches, method = self._semantic_search(
                anchor_idx, anchor_ngs, output_province, top_n
            )
        else:
            matches, method = self._jaccard_search(
                anchor_raw, anchor_ngs, output_province, top_n
            )

        return LookupResult(
            anchor          = anchor,
            output_province = output_province,
            matches         = matches,
            ngs_label       = ngs_label,
            ngs_description = ngs_desc,
            score_method    = method,
        )

    # ── semantic search (BGE embeddings) ─────────────────────────────────────

    def _semantic_search(
        self,
        anchor_idx: int,
        anchor_ngs: str,
        output_province: str,
        top_n: int,
    ) -> tuple[list[MatchResult], str]:

        # Row in embedding matrix for this anchor
        anchor_embed_row = int(np.where(self._embed_idx == anchor_idx)[0][0])
        q_vec = self._embed[anchor_embed_row]   # (D,) normalised

        # Candidate rows (output province)
        cand_record_indices = self._by_prov.get(output_province, [])
        if not cand_record_indices:
            return [], "semantic"

        # Map record indices -> embed rows
        # Build reverse map once (or on first use)
        if not hasattr(self, "_rec_to_embed"):
            self._rec_to_embed = {
                int(self._embed_idx[j]): j
                for j in range(len(self._embed_idx))
            }

        cand_embed_rows = np.array(
            [self._rec_to_embed[i] for i in cand_record_indices
             if i in self._rec_to_embed],
            dtype=np.int32,
        )
        if len(cand_embed_rows) == 0:
            return [], "semantic"

        cand_vecs  = self._embed[cand_embed_rows]         # (M, D)
        scores_raw = (cand_vecs @ q_vec).astype(float)    # cosine sim (already L2-normed)

        # --- ranking strategy ---
        # Rank purely by semantic similarity. Same-NGS is flagged informally
        # but does NOT boost or penalise scores — two provinces can legitimately
        # assign different NGS codes to equivalent procedures.
        use_ngs   = anchor_ngs and anchor_ngs not in ("NOMAP", "")
        ngs_flags = np.zeros(len(cand_embed_rows), dtype=bool)

        if use_ngs:
            for k, rec_i in enumerate(cand_record_indices):
                if rec_i not in self._rec_to_embed:
                    continue
                embed_row = self._rec_to_embed[rec_i]
                pos = np.where(cand_embed_rows == embed_row)[0]
                if len(pos) > 0:
                    ngs_flags[pos[0]] = (
                        self._records[rec_i].get("NGS_code") == anchor_ngs
                    )

        top_pos = np.argsort(-scores_raw)[:top_n]

        results = []
        for p in top_pos:
            rec_i    = cand_record_indices[p]
            rec      = self._records[rec_i]
            results.append(MatchResult(
                fee_code     = FeeCode.from_dict(rec),
                sim_score    = float(scores_raw[p]),
                ngs_match    = bool(ngs_flags[p]),   # informational only
                score_method = "semantic",
            ))

        return results, "semantic"

    # ── jaccard fallback ──────────────────────────────────────────────────────

    def _jaccard_search(
        self,
        anchor_raw: dict,
        anchor_ngs: str,
        output_province: str,
        top_n: int,
    ) -> tuple[list[MatchResult], str]:

        anchor_tok = _tokenize(_rec_text(anchor_raw))
        use_ngs    = anchor_ngs and anchor_ngs not in ("NOMAP", "")

        scored = []
        for i in self._by_prov.get(output_province, []):
            r  = self._records[i]
            sc = _jaccard_overlap(anchor_tok, _tokenize(_rec_text(r)))
            ngs_ok = use_ngs and r.get("NGS_code") == anchor_ngs
            scored.append((sc, ngs_ok, r))

        # Rank purely by similarity score; NGS match is informational only
        scored.sort(key=lambda x: -x[0])

        results = []
        for sc, ngs_ok, r in scored[:top_n]:
            results.append(MatchResult(
                fee_code     = FeeCode.from_dict(r),
                sim_score    = sc,
                ngs_match    = ngs_ok,
                score_method = "jaccard",
            ))
        return results, "jaccard"


# Singleton — reused across Streamlit reruns via st.cache_resource
_engine = LookupEngine()

def get_engine() -> LookupEngine:
    return _engine
