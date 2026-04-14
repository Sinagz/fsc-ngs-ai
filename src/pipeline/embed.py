"""OpenAI embeddings for canonical records. L2-normalized at write time."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from src.pipeline.schema import FeeCodeRecord


class _ClientLike(Protocol):
    def embed(self, texts: list[str], *, model: str, dim: int,
              batch_size: int = 500) -> list[list[float]]: ...


def _record_text(r: FeeCodeRecord) -> str:
    parts = [
        r.fsc_fn, r.fsc_description, r.fsc_section or "",
        r.fsc_subsection or "", r.fsc_chapter or "", r.NGS_label or "",
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


def build_embeddings(
    records: list[FeeCodeRecord], *, client: _ClientLike, model: str, dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build L2-normalized OpenAI embeddings for the given records.

    Returns (embeddings, record_ids) where:
    - embeddings is an (N, dim) float32 array with unit-norm rows
    - record_ids is an (N,) int32 array of positional indices into the input list

    IMPORTANT: record_ids are positional indices, not fsc_code lookups. The caller
    must persist `records` in the exact same order when reading the embeddings file;
    sorting or filtering `records` between build_embeddings() and consumers silently
    desynchronizes the .npz from the source data.
    """
    if not records:
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    texts = [_record_text(r) for r in records]
    vecs = client.embed(texts, model=model, dim=dim)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    ids = np.arange(len(records), dtype=np.int32)
    return arr, ids


def save_npz(path: Path, embeddings: np.ndarray, record_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), embeddings=embeddings, record_ids=record_ids)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(str(path), allow_pickle=False)
    return npz["embeddings"].astype(np.float32), npz["record_ids"].astype(np.int32)
