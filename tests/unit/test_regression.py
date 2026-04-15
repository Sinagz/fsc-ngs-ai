from src.pipeline.schema import FeeCodeRecord
from src.pipeline.regression import diff, check, DiffReport


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
