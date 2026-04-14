"""Regression diff + fail-loud gate."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.pipeline.schema import FeeCodeRecord, Province

PROVINCES: list[Province] = ["ON", "BC", "YT"]


@dataclass(frozen=True)
class DiffReport:
    before: dict[Province, int]
    after: dict[Province, int]
    added: dict[Province, int]
    removed: dict[Province, int]
    field_changed: dict[Province, int]
    removed_codes: set[tuple[Province, str]] = field(default_factory=set)


def _index(records: list[FeeCodeRecord]) -> dict[tuple[Province, str], FeeCodeRecord]:
    return {(r.province, r.fsc_code): r for r in records}


def _counts(records: list[FeeCodeRecord]) -> dict[Province, int]:
    out: dict[Province, int] = {p: 0 for p in PROVINCES}
    for r in records:
        out[r.province] += 1
    return out


def diff(*, new: list[FeeCodeRecord], old: list[FeeCodeRecord]) -> DiffReport:
    old_ix = _index(old)
    new_ix = _index(new)

    added: dict[Province, int] = defaultdict(int)
    removed: dict[Province, int] = defaultdict(int)
    changed: dict[Province, int] = defaultdict(int)
    removed_codes: set[tuple[Province, str]] = set()

    for key in new_ix.keys() - old_ix.keys():
        added[key[0]] += 1
    for key in old_ix.keys() - new_ix.keys():
        removed[key[0]] += 1
        removed_codes.add(key)
    for key in new_ix.keys() & old_ix.keys():
        if new_ix[key] != old_ix[key]:
            changed[key[0]] += 1

    return DiffReport(
        before=_counts(old),
        after=_counts(new),
        added={p: added[p] for p in PROVINCES},
        removed={p: removed[p] for p in PROVINCES},
        field_changed={p: changed[p] for p in PROVINCES},
        removed_codes=removed_codes,
    )


def check(
    report: DiffReport,
    *,
    golden_set: set[tuple[Province, str]],
    threshold: float = 0.05,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for p in PROVINCES:
        if report.before[p] == 0:
            continue
        drop = (report.before[p] - report.after[p]) / report.before[p]
        if drop >= threshold:
            reasons.append(
                f"{p}: code count drop {drop:.1%} exceeds {threshold:.0%} threshold"
            )
    missing_golden = report.removed_codes & golden_set
    if missing_golden:
        reasons.append(f"golden-set codes missing: {sorted(missing_golden)}")
    return (len(reasons) == 0, reasons)


def format_report(report: DiffReport) -> str:
    lines = ["Provinces:     ON      BC      YT"]
    lines.append(
        f"Before:        {report.before['ON']:<8}{report.before['BC']:<8}{report.before['YT']}"
    )
    lines.append(
        f"After:         {report.after['ON']:<8}{report.after['BC']:<8}{report.after['YT']}"
    )
    lines.append(
        f"Added:         +{report.added['ON']:<7}+{report.added['BC']:<7}+{report.added['YT']}"
    )
    lines.append(
        f"Removed:       -{report.removed['ON']:<7}-{report.removed['BC']:<7}-{report.removed['YT']}"
    )
    lines.append(
        f"Field-changed: +{report.field_changed['ON']:<7}+{report.field_changed['BC']:<7}+{report.field_changed['YT']}"
    )
    return "\n".join(lines)
