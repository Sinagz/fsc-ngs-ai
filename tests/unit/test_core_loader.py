import json
from pathlib import Path

import numpy as np
import pytest

from src.core.loader import load_latest
from src.pipeline.schema import FeeCodeRecord, Manifest


def _rec(province: str, code: str) -> FeeCodeRecord:
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description="d",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="vision", extraction_confidence=1.0,
    )


def _seed_version(parsed_dir: Path, version: str, with_embeddings: bool = True) -> None:
    vdir = parsed_dir / f"v{version}"
    vdir.mkdir(parents=True)
    recs = [_rec("ON", "K040")]
    (vdir / "codes.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in recs])
    )
    if with_embeddings:
        np.savez(
            str(vdir / "embeddings.npz"),
            embeddings=np.zeros((1, 2), dtype=np.float32),
            record_ids=np.array([0], dtype=np.int32),
        )
    (vdir / "manifest.json").write_text(
        Manifest(
            generated_at="t", git_sha="x",
            row_counts={"ON": 1, "BC": 0, "YT": 0},
            source_pdf_hashes={}, models={},
        ).model_dump_json()
    )


def test_load_latest_picks_max_version(tmp_path: Path):
    _seed_version(tmp_path, "2026-01-01")
    _seed_version(tmp_path, "2026-04-13")
    records, emb, ids, manifest = load_latest(tmp_path)
    assert manifest.row_counts["ON"] == 1
    assert len(records) == 1
    assert records[0].fsc_code == "K040"
    assert emb.shape == (1, 2)
    assert ids.shape == (1,)


def test_load_latest_handles_missing_embeddings(tmp_path: Path):
    _seed_version(tmp_path, "2026-04-13", with_embeddings=False)
    records, emb, ids, manifest = load_latest(tmp_path)
    assert len(records) == 1
    assert emb.size == 0
    assert ids.size == 0


def test_load_latest_raises_when_no_versions(tmp_path: Path):
    (tmp_path / "not-a-version").mkdir()
    with pytest.raises(FileNotFoundError):
        load_latest(tmp_path)


def test_loader_rejects_v1_artifacts(tmp_path):
    vdir = tmp_path / "v2025-01-01"
    vdir.mkdir()
    (vdir / "codes.json").write_text(json.dumps([
        {
            "schema_version": "1",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "y",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "structural",
            "extraction_confidence": 0.9,
        }
    ]))
    (vdir / "manifest.json").write_text(json.dumps({
        "schema_version": "1", "generated_at": "2025-01-01T00:00:00+00:00",
        "git_sha": "abc", "row_counts": {"ON": 1, "BC": 0, "YT": 0},
        "source_pdf_hashes": {"ON": "h"}, "models": {"embed": "x", "extract": "y"},
        "regression_override": None,
    }))
    np.savez(vdir / "embeddings.npz",
             embeddings=np.zeros((1, 1024), dtype=np.float32),
             record_ids=np.array([0], dtype=np.int32))

    with pytest.raises(RuntimeError, match="schema v1.*rerun the pipeline"):
        load_latest(tmp_path)
