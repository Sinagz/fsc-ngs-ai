"""Snapshot regression test for the latest pipeline output.

First run seeds the snapshot; subsequent runs assert that no
golden code disappears and that no more than 5% of new codes
appear without explicit re-seeding.

To re-seed after an intentional change: delete `tests/regression/snapshot_codes.json`
and re-run pytest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PARSED_DIR = ROOT / "data" / "parsed"
SNAPSHOT = ROOT / "tests" / "regression" / "snapshot_codes.json"


@pytest.mark.regression
def test_codes_match_snapshot():
    version_dirs = sorted(PARSED_DIR.glob("v*"))
    if not version_dirs:
        pytest.skip(
            f"No versioned pipeline artifacts under {PARSED_DIR}; "
            "run `python -m src.cli run` first"
        )
    current = json.loads(
        (version_dirs[-1] / "codes.json").read_text(encoding="utf-8")
    )

    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(
            json.dumps(current, indent=2), encoding="utf-8"
        )
        pytest.skip("Snapshot created; re-run to compare against future runs")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    expected_keys = {(r["province"], r["fsc_code"]) for r in expected}
    current_keys = {(r["province"], r["fsc_code"]) for r in current}

    missing = sorted(expected_keys - current_keys)
    added = sorted(current_keys - expected_keys)

    assert not missing, (
        f"{len(missing)} codes removed vs snapshot (sample): {missing[:10]}"
    )
    # Allow up to 5% growth without re-seeding; large additions = re-seed signal
    if expected_keys:
        growth = len(added) / len(expected_keys)
        assert growth < 0.05, (
            f"too many new codes vs snapshot ({len(added)} new, "
            f"{growth:.1%} growth); delete snapshot_codes.json to re-seed"
        )
