"""
Build semantic embeddings for all fee codes using BAAI/bge-large-en-v1.5.

Why bge-large-en-v1.5:
  - Best-in-class retrieval quality for English text
  - 1024-dim embeddings, ~1.3 GB on GPU — fits comfortably in 4 GB VRAM
  - Outperforms smaller models on asymmetric and symmetric search tasks
  - No API key, runs fully local on CUDA

Output:
  data/parsed/embeddings.npz
    - 'embeddings'  : float32 array (N, 1024), L2-normalised
    - 'record_ids'  : int array (N,) — index into fsc_ngs_mapped.json
    - 'provinces'   : str array (N,)
    - 'fsc_codes'   : str array (N,)

Run:
    python src/build_embeddings.py
Re-run after any pipeline update that changes fsc_ngs_mapped.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR    = Path(__file__).resolve().parent.parent
MAPPED_PATH = BASE_DIR / "data" / "parsed" / "fsc_ngs_mapped.json"
OUT_PATH    = BASE_DIR / "data" / "parsed" / "embeddings.npz"

MODEL_NAME  = "BAAI/bge-large-en-v1.5"
BATCH_SIZE  = 128   # safe for 4 GB VRAM with bge-large


def build_text(r: dict) -> str:
    """
    Construct a rich text representation of one fee code record.
    Concatenates all meaningful fields so the embedding captures
    clinical meaning, chapter context, and section hierarchy.
    """
    parts = []
    if r.get("fsc_fn"):
        parts.append(r["fsc_fn"])
    if r.get("fsc_description"):
        parts.append(r["fsc_description"])
    if r.get("fsc_section"):
        parts.append(r["fsc_section"])
    if r.get("fsc_subsection"):
        parts.append(r["fsc_subsection"])
    if r.get("fsc_chapter"):
        parts.append(r["fsc_chapter"])
    if r.get("NGS_label"):
        parts.append(r["NGS_label"])
    return " | ".join(p.strip() for p in parts if p.strip()) or r.get("fsc_code", "")


def main():
    print(f"Loading fee codes from {MAPPED_PATH.name}...", flush=True)
    records = json.loads(MAPPED_PATH.read_text(encoding="utf-8"))
    print(f"  {len(records)} records")

    texts     = [build_text(r) for r in records]
    provinces = np.array([r["province"] for r in records])
    fsc_codes = np.array([r["fsc_code"] for r in records])
    record_ids = np.arange(len(records), dtype=np.int32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model: {MODEL_NAME}  (device={device})", flush=True)
    if device == "cuda":
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  VRAM free before load: {vram_free:.2f} GB")

    model = SentenceTransformer(MODEL_NAME, device=device)

    if device == "cuda":
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  VRAM free after load:  {vram_free:.2f} GB")

    print(f"\nEncoding {len(texts)} texts  (batch_size={BATCH_SIZE})...", flush=True)
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # L2-norm → cosine sim = dot product
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  — shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    np.savez_compressed(
        str(OUT_PATH),
        embeddings  = embeddings.astype(np.float32),
        record_ids  = record_ids,
        provinces   = provinces,
        fsc_codes   = fsc_codes,
    )
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved -> {OUT_PATH}  ({size_mb:.1f} MB)")

    # Sanity check
    loaded = np.load(str(OUT_PATH), allow_pickle=False)
    print(f"\nSanity check — shapes:")
    for k in loaded.files:
        print(f"  {k}: {loaded[k].shape}")


if __name__ == "__main__":
    main()
