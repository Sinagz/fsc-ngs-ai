from decimal import Decimal

from src.pipeline.schema import FeeCodeRecord
from src.pipeline.regression import check, check_golden_set_invariants, diff, DiffReport


def _rec(code: str, province="ON", desc="desc"):
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description=desc,
        page=1, source_pdf_hash="a" * 64,
        extraction_method="vision", extraction_confidence=1.0,
    )


def test_diff_counts_added_removed_changed():
    old = [_rec("K040"), _rec("K041"), _rec("01712", province="BC")]
    new = [_rec("K040"), _rec("K042"), _rec("01712", province="BC", desc="new")]
    report = diff(new=new, old=old)
    assert report.added == {"ON": 1, "BC": 0, "YT": 0}
    assert report.removed == {"ON": 1, "BC": 0, "YT": 0}
    assert report.field_changed == {"ON": 0, "BC": 1, "YT": 0}


def test_check_fails_on_large_drop():
    old = [_rec(f"K{i:03d}") for i in range(100)]
    new = [_rec(f"K{i:03d}") for i in range(90)]  # 10% drop
    report = diff(new=new, old=old)
    ok, reasons = check(report, golden_set=set(), threshold=0.05)
    assert not ok
    assert any("drop" in r.lower() for r in reasons)


def test_check_fails_on_missing_golden_code():
    old = [_rec("K040")]
    new: list = []
    report = diff(new=new, old=old)
    ok, reasons = check(report, golden_set={("ON", "K040")}, threshold=0.05)
    assert not ok
    assert any("golden" in r.lower() for r in reasons)


def test_check_passes_on_small_change():
    old = [_rec(f"K{i:03d}") for i in range(100)]
    new = [_rec(f"K{i:03d}") for i in range(100)] + [_rec("K999")]
    report = diff(new=new, old=old)
    ok, reasons = check(report, golden_set=set(), threshold=0.05)
    assert ok
    assert reasons == []


# ---------------------------------------------------------------------------
# check_golden_set_invariants tests (Task 16)
# ---------------------------------------------------------------------------

def _grec(code: str, price: str | None, desc: str, province: str = "ON") -> FeeCodeRecord:
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description=desc,
        page=1, source_pdf_hash="h",
        price=Decimal(price) if price else None,
        extraction_method="vision", extraction_confidence=0.9,
    )


def test_golden_invariants_pass_when_all_fields_match():
    records = [_grec("E611", "82.75", "general consultation and specific assessment")]
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": "82.75",
        "expected_description_contains": ["consultation", "assessment"],
    }]
    assert check_golden_set_invariants(records=records, golden=golden) == []


def test_golden_invariants_flag_price_mismatch():
    records = [_grec("E611", "7.00", "general consultation and assessment")]
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": "82.75",
        "expected_description_contains": ["consultation"],
    }]
    issues = check_golden_set_invariants(records=records, golden=golden)
    assert any("price" in i.lower() for i in issues)
    assert any("82.75" in i for i in issues)
    assert any("7.00" in i for i in issues)


def test_golden_invariants_flag_missing_substring():
    records = [_grec("E611", "82.75", "something unrelated entirely")]
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": "82.75",
        "expected_description_contains": ["consultation", "assessment"],
    }]
    issues = check_golden_set_invariants(records=records, golden=golden)
    assert any("consultation" in i for i in issues)
    assert any("assessment" in i for i in issues)


def test_golden_invariants_flag_missing_record():
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": "82.75",
    }]
    issues = check_golden_set_invariants(records=[], golden=golden)
    assert any("E611" in i for i in issues)


def test_golden_invariants_tolerates_missing_expected_price():
    """When expected_price is absent, price check is skipped (not asserted)."""
    records = [_grec("E611", "7.00", "general consultation")]
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_description_contains": ["consultation"],
    }]
    # Price is intentionally not checked -> passes despite the "7.00" that would fail a strict check
    assert check_golden_set_invariants(records=records, golden=golden) == []


def test_golden_invariants_expected_price_none_requires_none():
    """expected_price=None means the record must also have None price."""
    records = [_grec("E611", "25.50", "general consultation")]
    golden_null_matching = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": None,
    }]
    issues = check_golden_set_invariants(records=records, golden=golden_null_matching)
    assert any("price" in i.lower() for i in issues)

    records_null = [_grec("E611", None, "general consultation")]
    assert check_golden_set_invariants(records=records_null, golden=golden_null_matching) == []


def test_golden_invariants_tolerates_missing_description_field():
    records = [_grec("E611", "82.75", "whatever")]
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": "82.75",
    }]
    assert check_golden_set_invariants(records=records, golden=golden) == []


def test_golden_invariants_case_insensitive_description():
    records = [_grec("E611", "82.75", "General CONSULTATION provided")]
    golden = [{
        "province": "ON", "fsc_code": "E611",
        "expected_price": "82.75",
        "expected_description_contains": ["consultation", "provided"],
    }]
    assert check_golden_set_invariants(records=records, golden=golden) == []
