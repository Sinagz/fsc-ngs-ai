from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.pipeline.schema import FeeCodeRecord
from src.pipeline.embed import build_embeddings, save_npz, load_npz


def _rec(code: str) -> FeeCodeRecord:
    return FeeCodeRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="desc",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
    )


def test_build_embeddings_returns_l2_normalized():
    recs = [_rec("K040"), _rec("K041")]
    client = MagicMock()
    client.embed.return_value = [[3.0, 4.0], [1.0, 0.0]]
    arr, ids = build_embeddings(recs, client=client, model="m", dim=2)
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(arr, axis=1), [1.0, 1.0])
    np.testing.assert_array_equal(ids, [0, 1])


def test_npz_roundtrip(tmp_path: Path):
    arr = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([7, 42], dtype=np.int32)
    out = tmp_path / "e.npz"
    save_npz(out, arr, ids)
    arr2, ids2 = load_npz(out)
    np.testing.assert_array_equal(arr, arr2)
    np.testing.assert_array_equal(ids, ids2)
