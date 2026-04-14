"""Golden-set integration test.

Asserts that the 60 hand-curated codes (20 per province) survive a real pipeline
run and retain their expected description substrings. Skipped when no versioned
artifacts are present, since this test reads the pipeline's actual output.

To populate: run `python -m src.cli run` (requires raw PDFs in data/raw/pdf/
and a valid OPENAI_API_KEY in .env).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.loader import load_latest

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "golden_codes.json"
PARSED_DIR = ROOT / "data" / "parsed"


def _versioned_dirs() -> list[Path]:
    return sorted(PARSED_DIR.glob("v*"))


@pytest.mark.integration
def test_golden_set_codes_all_present():
    if not _versioned_dirs():
        pytest.skip(
            f"No versioned pipeline artifacts under {PARSED_DIR}; "
            "run `python -m src.cli run` first"
        )
    records, _, _, _ = load_latest(PARSED_DIR)
    index = {(r.province, r.fsc_code): r for r in records}

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    missing = [
        (g["province"], g["fsc_code"])
        for g in golden
        if (g["province"], g["fsc_code"]) not in index
    ]
    assert not missing, (
        f"Missing {len(missing)} of {len(golden)} golden-set codes: "
        f"{missing[:10]}"
    )


@pytest.mark.integration
def test_golden_set_descriptions_contain_expected_substrings():
    if not _versioned_dirs():
        pytest.skip("No versioned pipeline artifacts; see test_golden_set_codes_all_present")
    records, _, _, _ = load_latest(PARSED_DIR)
    index = {(r.province, r.fsc_code): r for r in records}

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for g in golden:
        key = (g["province"], g["fsc_code"])
        if key not in index:
            continue  # covered by the previous test
        r = index[key]
        desc = (r.fsc_description or "").lower()
        for needle in g.get("expected_description_contains", []):
            if needle.lower() not in desc:
                failures.append(
                    f"{g['province']}:{g['fsc_code']} missing '{needle}' "
                    f"in {desc[:80]!r}"
                )
    assert not failures, f"{len(failures)} description checks failed:\n" + "\n".join(failures[:10])
