"""Resolve FSC codes to NGS categories.

Three tiers: exact code_refs match, semantic top-1 with LLM verification,
NOMAP. Output order mirrors input order regardless of which tier resolved
each row, so downstream positional joins (e.g. embeddings.npz row indices)
remain valid.

Error handling asymmetry:

* Client errors raised by ``chat_json`` are caught and the affected row
  is routed to NOMAP — a single transient LLM failure does not abort the
  whole batch.
* Errors raised by ``embed`` are **not** caught. Embedding operates on
  all unresolved rows at once, so a failure there affects every row
  equally and should propagate for the caller to handle.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

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
    """Map each ``FeeCodeRecord`` to an ``NGSRecord`` via three tiers.

    Order of the returned list matches the order of ``records``.

    Client errors in ``chat_json`` are caught and the row is routed to
    NOMAP; embedding failures are not caught (they affect all
    unresolved rows, so they propagate).
    """
    # 1. Exact-match pass
    exact_index: dict[str, NGSRecord] = {}
    for n in ngs:
        for ref in n.code_refs:
            exact_index.setdefault(ref.upper(), n)

    out: list[FeeCodeRecord | None] = [None] * len(records)
    unresolved_slots: list[int] = []
    unresolved: list[FeeCodeRecord] = []

    for i, r in enumerate(records):
        hit = exact_index.get(r.fsc_code.upper())
        if hit is not None:
            out[i] = r.model_copy(update={
                "NGS_code": hit.ngs_code,
                "NGS_label": hit.ngs_label,
                "NGS_mapping_method": "exact",
                "NGS_mapping_confidence": 1.0,
            })
        else:
            unresolved_slots.append(i)
            unresolved.append(r)

    if not unresolved or not ngs:
        # Fill any remaining slots with their original unresolved records
        for slot, r in zip(unresolved_slots, unresolved):
            out[slot] = r
        return [r for r in out if r is not None]

    # 2. Semantic pass
    fee_texts = [f"{r.fsc_fn}. {r.fsc_description}" for r in unresolved]
    ngs_texts = [f"{n.ngs_label}. {n.ngs_description}" for n in ngs]
    vecs = client.embed(fee_texts + ngs_texts, model=embed_model, dim=dim)
    arr = _normalize(vecs)
    fee_vecs = arr[: len(fee_texts)]
    ngs_vecs = arr[len(fee_texts):]
    sims = fee_vecs @ ngs_vecs.T  # (F, N)

    iterator = tqdm(
        enumerate(zip(unresolved_slots, unresolved)),
        total=len(unresolved),
        desc="NGS LLM verdicts",
        unit="code",
        leave=False,
    )
    for i, (slot, r) in iterator:
        top_j = int(np.argmax(sims[i]))
        top_sim = float(sims[i, top_j])
        if top_sim < SIMILARITY_THRESHOLD:
            out[slot] = r
            continue
        candidate = ngs[top_j]
        try:
            verdict = client.chat_json(
                prompt=(
                    f"Fee code {r.fsc_code} ({r.province}): {r.fsc_fn}. {r.fsc_description}\n"
                    f"Candidate NGS category {candidate.ngs_code}: "
                    f"{candidate.ngs_label}. {candidate.ngs_description}\n"
                    "Is this the correct category?"
                ),
                schema=NGSVerdict, model=llm_model, temperature=0.0,
            )
        except Exception:  # noqa: BLE001 — treat as NOMAP, keep pipeline running
            out[slot] = r
            continue

        if verdict.accept:
            out[slot] = r.model_copy(update={
                "NGS_code": candidate.ngs_code,
                "NGS_label": candidate.ngs_label,
                "NGS_mapping_method": "llm",
                "NGS_mapping_confidence": verdict.confidence,
            })
        else:
            out[slot] = r

    return [r for r in out if r is not None]
