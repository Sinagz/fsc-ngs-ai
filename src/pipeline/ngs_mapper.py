"""Resolve FSC codes to NGS categories.
Three tiers: exact code_refs match, semantic top-1 with LLM verification, NOMAP."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from src.pipeline.schema import FeeCodeRecord, NGSRecord

SIMILARITY_THRESHOLD = 0.5


class NGSVerdict(BaseModel):
    accept: bool
    confidence: float = Field(ge=0.0, le=1.0)


class _ClientLike(Protocol):
    def embed(self, texts: list[str], *, model: str, dim: int,
              batch_size: int = 500) -> list[list[float]]: ...
    def chat_json(self, *, prompt: str, schema: type, model: str,
                  system: str | None = None, temperature: float = 0.0): ...


def _normalize(vecs: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def map_ngs(
    records: list[FeeCodeRecord],
    ngs: list[NGSRecord],
    *,
    client: _ClientLike,
    embed_model: str,
    llm_model: str,
    dim: int,
) -> list[FeeCodeRecord]:
    # 1. Exact-match pass
    exact_index: dict[str, NGSRecord] = {}
    for n in ngs:
        for ref in n.code_refs:
            exact_index.setdefault(ref.upper(), n)

    out: list[FeeCodeRecord] = []
    unresolved: list[FeeCodeRecord] = []
    for r in records:
        hit = exact_index.get(r.fsc_code.upper())
        if hit is not None:
            out.append(r.model_copy(update={
                "NGS_code": hit.ngs_code,
                "NGS_label": hit.ngs_label,
                "NGS_mapping_method": "exact",
                "NGS_mapping_confidence": 1.0,
            }))
        else:
            unresolved.append(r)

    if not unresolved or not ngs:
        return out + unresolved

    # 2. Semantic pass
    fee_texts = [f"{r.fsc_fn}. {r.fsc_description}" for r in unresolved]
    ngs_texts = [f"{n.ngs_label}. {n.ngs_description}" for n in ngs]
    vecs = client.embed(fee_texts + ngs_texts, model=embed_model, dim=dim)
    arr = _normalize(vecs)
    fee_vecs = arr[: len(fee_texts)]
    ngs_vecs = arr[len(fee_texts):]
    sims = fee_vecs @ ngs_vecs.T  # (F, N)

    for i, r in enumerate(unresolved):
        top_j = int(np.argmax(sims[i]))
        top_sim = float(sims[i, top_j])
        if top_sim < SIMILARITY_THRESHOLD:
            out.append(r)
            continue
        candidate = ngs[top_j]
        verdict = client.chat_json(
            prompt=(
                f"Fee code {r.fsc_code} ({r.province}): {r.fsc_fn}. {r.fsc_description}\n"
                f"Candidate NGS category {candidate.ngs_code}: "
                f"{candidate.ngs_label}. {candidate.ngs_description}\n"
                "Is this the correct category?"
            ),
            schema=NGSVerdict, model=llm_model, temperature=0.0,
        )
        if verdict.accept:
            out.append(r.model_copy(update={
                "NGS_code": candidate.ngs_code,
                "NGS_label": candidate.ngs_label,
                "NGS_mapping_method": "llm",
                "NGS_mapping_confidence": verdict.confidence,
            }))
        else:
            out.append(r)
    return out
