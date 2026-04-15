"""Hypothesis property tests for the canonical pydantic schema."""
from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings, strategies as st

from src.pipeline.schema import FeeCodeRecord

VALID_PROVINCE = st.sampled_from(["ON", "BC", "YT"])
ALPHANUM = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=3,
    max_size=5,
)
CONFIDENCE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
PRICE = st.one_of(
    st.none(),
    st.decimals(
        min_value=Decimal("0.01"),
        max_value=Decimal("99999.99"),
        places=2,
        allow_nan=False,
        allow_infinity=False,
    ),
)


@pytest.mark.property
@settings(max_examples=100)
@given(
    province=VALID_PROVINCE,
    code=ALPHANUM,
    page=st.integers(min_value=1, max_value=2000),
    confidence=CONFIDENCE,
    price=PRICE,
)
def test_record_json_roundtrip(province, code, page, confidence, price):
    """JSON dump then re-instantiate must produce an equal record."""
    r = FeeCodeRecord(
        province=province,
        fsc_code=code,
        fsc_fn="fn",
        fsc_description="d",
        page=page,
        source_pdf_hash="a" * 64,
        extraction_method="vision",
        extraction_confidence=confidence,
        price=price,
    )
    data = r.model_dump(mode="json")
    r2 = FeeCodeRecord(**data)
    assert r == r2


@pytest.mark.property
@settings(max_examples=50)
@given(province=VALID_PROVINCE, code=ALPHANUM, conf=CONFIDENCE)
def test_record_is_immutable(province, code, conf):
    """Frozen dataclass-like behavior under hypothesis: mutation always raises."""
    from pydantic import ValidationError

    r = FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description="d",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="vision", extraction_confidence=conf,
    )
    with pytest.raises(ValidationError):
        r.fsc_code = "X"  # type: ignore[misc]
