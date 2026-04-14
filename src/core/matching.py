"""Phase 1: ported cosine-similarity + Jaccard-fallback matching.

Same algorithm as the pre-rebuild app/lookup_engine.py but reading the new
versioned artifacts and the canonical FeeCodeRecord schema. Phase 2 will
replace this with retrieval + LLM reranking; the search() signature stays
stable across that change (Contract B in the design spec).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.pipeline.schema import FeeCodeRecord, Province

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "at", "from", "as", "is", "are", "be", "this", "that", "per", "each",
    "full", "quarter", "hour", "payment", "rules", "including", "only", "when",
    "not", "no", "any", "all", "if", "other", "than", "also", "where", "may",
    "used", "use", "within", "after", "before", "without", "during", "which",
    "who", "same", "more", "one", "two", "three", "four", "five",
})


@dataclass(frozen=True)
class MatchResult:
    fee_code: FeeCodeRecord
    sim_score: float
    ngs_match: bool
    score_method: str  # "semantic" | "jaccard"


@dataclass(frozen=True)
class LookupResult:
    anchor: FeeCodeRecord
    output_province: Province
    matches: list[MatchResult]
    score_method: str


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {t for t in text.split() if t and t not in STOPWORDS and len(t) > 2}


def _jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    j = inter / len(a | b)
    o = inter / max(len(a), 1)
    return round(0.4 * j + 0.6 * o, 4)


def _record_text(r: FeeCodeRecord) -> str:
    parts = [
        r.fsc_fn, r.fsc_description, r.fsc_section or "",
        r.fsc_subsection or "", r.fsc_chapter or "", r.NGS_label or "",
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


def _ngs_match(a: FeeCodeRecord, b: FeeCodeRecord) -> bool:
    """Same NGS code, ignoring missing/NOMAP."""
    return bool(
        a.NGS_code
        and a.NGS_code not in ("NOMAP", "")
        and a.NGS_code == b.NGS_code
    )


def search(
    *,
    fsc_code: str,
    src: Province,
    dst: Province,
    top_n: int,
    records: list[FeeCodeRecord],
    embeddings: np.ndarray,
    record_ids: np.ndarray,
) -> LookupResult | None:
    """Look up `fsc_code` in `src` province and return top-N matches in `dst`.

    Uses semantic cosine similarity if `embeddings` is non-empty; otherwise
    falls back to Jaccard+overlap on tokenized descriptions. Same-NGS is
    surfaced as an informational flag but does NOT boost ranking — two
    provinces can legitimately assign equivalent procedures to different NGS
    codes (see design spec section 4).

    Returns None if `fsc_code` is not found in `src`.
    """
    code = fsc_code.strip().upper()
    anchor_idx = next(
        (i for i, r in enumerate(records)
         if r.province == src and r.fsc_code.upper() == code),
        None,
    )
    if anchor_idx is None:
        return None
    anchor = records[anchor_idx]

    candidate_indices = [i for i, r in enumerate(records) if r.province == dst]
    if not candidate_indices:
        return LookupResult(
            anchor=anchor, output_province=dst, matches=[],
            score_method="semantic" if embeddings.size > 0 else "jaccard",
        )

    has_embed = embeddings.size > 0 and len(record_ids) == len(records)
    if has_embed:
        matches = _semantic_search(
            anchor_idx, anchor, candidate_indices, records,
            embeddings, record_ids, top_n,
        )
        method = "semantic"
    else:
        matches = _jaccard_search(anchor, candidate_indices, records, top_n)
        method = "jaccard"
    return LookupResult(
        anchor=anchor, output_province=dst, matches=matches, score_method=method,
    )


def _semantic_search(
    anchor_idx: int,
    anchor: FeeCodeRecord,
    cand_indices: list[int],
    records: list[FeeCodeRecord],
    embeddings: np.ndarray,
    record_ids: np.ndarray,
    top_n: int,
) -> list[MatchResult]:
    rec_to_embed = {int(record_ids[j]): j for j in range(len(record_ids))}
    if anchor_idx not in rec_to_embed:
        return []
    q = embeddings[rec_to_embed[anchor_idx]]
    cand_rows = np.array(
        [rec_to_embed[i] for i in cand_indices if i in rec_to_embed],
        dtype=np.int32,
    )
    if cand_rows.size == 0:
        return []
    cand_vecs = embeddings[cand_rows]
    scores = (cand_vecs @ q).astype(float)
    top = np.argsort(-scores)[:top_n]
    return [
        MatchResult(
            fee_code=records[cand_indices[p]],
            sim_score=float(scores[p]),
            ngs_match=_ngs_match(anchor, records[cand_indices[p]]),
            score_method="semantic",
        )
        for p in top
    ]


def _jaccard_search(
    anchor: FeeCodeRecord,
    cand_indices: list[int],
    records: list[FeeCodeRecord],
    top_n: int,
) -> list[MatchResult]:
    q_tok = _tokenize(_record_text(anchor))
    scored = [
        (_jaccard_overlap(q_tok, _tokenize(_record_text(records[i]))), records[i])
        for i in cand_indices
    ]
    scored.sort(key=lambda x: -x[0])
    return [
        MatchResult(
            fee_code=r, sim_score=sc, ngs_match=_ngs_match(anchor, r),
            score_method="jaccard",
        )
        for sc, r in scored[:top_n]
    ]
