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
