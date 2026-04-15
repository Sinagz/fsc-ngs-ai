"""Load the newest versioned pipeline artifact bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.schema import FeeCodeRecord, Manifest


def load_latest(
    parsed_dir: Path,
) -> tuple[list[FeeCodeRecord], np.ndarray, np.ndarray, Manifest]:
    """Return (records, embeddings, record_ids, manifest) from the newest v* dir.

    Embeddings are optional — if `embeddings.npz` is missing, returns zero-shape
    arrays so callers can fall back to a non-semantic search path.
    """
    version_dirs = sorted(parsed_dir.glob("v*"))
    if not version_dirs:
        raise FileNotFoundError(f"No versioned artifacts in {parsed_dir}")
    vdir = version_dirs[-1]

    codes_raw = json.loads((vdir / "codes.json").read_text(encoding="utf-8"))

    # Schema-version guard: refuse v1 up front with an actionable message.
    # pydantic would also reject, but the error message would be cryptic.
    if codes_raw and codes_raw[0].get("schema_version") != "2":
        raise RuntimeError(
            f"Artifact at {vdir} uses schema v{codes_raw[0].get('schema_version')}; "
            "this app expects v2. rerun the pipeline: python -m src.cli run --force"
        )

    records = [FeeCodeRecord(**c) for c in codes_raw]

    embeddings = np.zeros((0, 0), dtype=np.float32)
    record_ids = np.zeros((0,), dtype=np.int32)
    emb_path = vdir / "embeddings.npz"
    if emb_path.exists():
        npz = np.load(str(emb_path), allow_pickle=False)
        embeddings = npz["embeddings"].astype(np.float32)
        record_ids = npz["record_ids"].astype(np.int32)

    manifest = Manifest.model_validate_json(
        (vdir / "manifest.json").read_text(encoding="utf-8")
    )
    return records, embeddings, record_ids, manifest
