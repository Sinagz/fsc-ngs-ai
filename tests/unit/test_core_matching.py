import numpy as np

from src.core.matching import LookupResult, search
from src.pipeline.schema import FeeCodeRecord


def _rec(province: str, code: str, desc: str, ngs_code: str | None = "1AA") -> FeeCodeRecord:
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn=code, fsc_description=desc,
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
        NGS_code=ngs_code, NGS_label="Exam" if ngs_code else None,
    )


def test_search_returns_anchor_and_matches_semantic():
    records = [
        _rec("ON", "K040", "periodic health visit adult"),
        _rec("BC", "01712", "periodic health visit adult consultation"),
        _rec("BC", "02000", "fracture repair leg"),
    ]
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    record_ids = np.array([0, 1, 2], dtype=np.int32)
    result = search(
        fsc_code="K040", src="ON", dst="BC", top_n=2,
        records=records, embeddings=embeddings, record_ids=record_ids,
    )
    assert isinstance(result, LookupResult)
    assert result.anchor.fsc_code == "K040"
    assert result.score_method == "semantic"
    assert result.matches[0].fee_code.fsc_code == "01712"
    assert result.matches[0].sim_score > result.matches[1].sim_score


def test_search_falls_back_to_jaccard_when_no_embeddings():
    records = [
        _rec("ON", "K040", "periodic health visit adult"),
        _rec("BC", "01712", "periodic health visit adult consultation"),
    ]
    result = search(
        fsc_code="K040", src="ON", dst="BC", top_n=1,
        records=records,
        embeddings=np.zeros((0, 0), dtype=np.float32),
        record_ids=np.zeros((0,), dtype=np.int32),
    )
    assert result is not None
    assert result.score_method == "jaccard"
    assert result.matches[0].fee_code.fsc_code == "01712"
    assert 0.0 < result.matches[0].sim_score <= 1.0


def test_search_missing_code_returns_none():
    records = [_rec("ON", "K040", "x")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    record_ids = np.array([0], dtype=np.int32)
    result = search(
        fsc_code="NOPE", src="ON", dst="BC", top_n=5,
        records=records, embeddings=embeddings, record_ids=record_ids,
    )
    assert result is None


def test_search_no_candidates_returns_empty_matches():
    records = [_rec("ON", "K040", "x")]
    result = search(
        fsc_code="K040", src="ON", dst="BC", top_n=5, records=records,
        embeddings=np.zeros((0, 0), dtype=np.float32),
        record_ids=np.zeros((0,), dtype=np.int32),
    )
    assert result is not None
    assert result.matches == []


def test_ngs_match_flag_true_only_when_codes_equal_and_present():
    records = [
        _rec("ON", "K040", "x", ngs_code="1AA"),
        _rec("BC", "01712", "x", ngs_code="1AA"),
        _rec("BC", "02000", "x", ngs_code="1BB"),
        _rec("BC", "03000", "x", ngs_code=None),
    ]
    embeddings = np.array(
        [[1, 0], [1, 0], [0.9, 0.1], [0.8, 0.1]], dtype=np.float32
    )
    record_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    result = search(
        fsc_code="K040", src="ON", dst="BC", top_n=3,
        records=records, embeddings=embeddings, record_ids=record_ids,
    )
    assert result is not None
    by_code = {m.fee_code.fsc_code: m for m in result.matches}
    assert by_code["01712"].ngs_match is True
    assert by_code["02000"].ngs_match is False
    assert by_code["03000"].ngs_match is False
