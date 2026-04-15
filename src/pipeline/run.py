"""Pipeline orchestrator. Idempotent (skip-if-output-exists), --force to redo."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.openai_client import OpenAIClient
from src.pipeline.embed import build_embeddings, save_npz
from src.pipeline.ngs_mapper import map_ngs
from src.pipeline.ngs_parser import parse_ngs_docx
from src.pipeline.regression import check, diff, format_report
from src.pipeline.schema import FeeCodeRecord, Manifest
from src.pipeline.validate import validate
from src.pipeline.vision import extract_province as vision_extract_province

logger = logging.getLogger(__name__)


def _tokens(snapshot: dict[str, dict[str, int]]) -> int:
    return sum(s["prompt_tokens"] + s["completion_tokens"] for s in snapshot.values())


@dataclass(frozen=True)
class PipelineConfig:
    raw_pdf_dir: Path
    raw_docx_dir: Path
    output_dir: Path
    diagnostics_dir: Path
    version: str
    force: bool = False
    accept_regression: str | None = None
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 1024
    extract_model: str = "gpt-5.4-mini"
    golden_set: frozenset[tuple[str, str]] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PipelineResult:
    skipped: bool
    version_dir: Path
    manifest: Manifest | None


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _province_pdf(pdf_dir: Path, province: str) -> Path | None:
    patterns = {
        "ON": "moh-schedule-benefit-*.pdf",
        "BC": "msc_payment_schedule_*.pdf",
        "YT": "yukon_physician_fee_guide_*.pdf",
    }
    hits = sorted(pdf_dir.glob(patterns[province]))
    return hits[-1] if hits else None


def _pdf_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(
    cfg: PipelineConfig, *, client: OpenAIClient | None
) -> PipelineResult:
    version_dir = cfg.output_dir / f"v{cfg.version}"
    codes_path = version_dir / "codes.json"
    embeddings_path = version_dir / "embeddings.npz"
    manifest_path = version_dir / "manifest.json"

    if not cfg.force and codes_path.exists() and manifest_path.exists():
        logger.info("Skipping: artifacts already at %s (use --force to redo)", version_dir)
        return PipelineResult(skipped=True, version_dir=version_dir, manifest=None)

    if client is None:
        raise RuntimeError("OpenAIClient required when pipeline is not skipping")

    version_dir.mkdir(parents=True, exist_ok=True)
    cfg.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Clear diagnostics from any prior run so a --force re-run doesn't silently
    # concatenate this run's output with stale rows.
    # (window_failures.jsonl is unlinked in Phase 1 directly.)

    t0 = time.monotonic()
    logger.info("Pipeline started (version=%s, force=%s)", cfg.version, cfg.force)

    # Phase 1: Vision extraction per province
    logger.info("Phase 1/5: Vision extraction (per province)")
    phase_t0 = time.monotonic()
    records: list[FeeCodeRecord] = []
    pdf_hashes: dict[str, str] = {}
    failure_log = cfg.diagnostics_dir / "window_failures.jsonl"
    failure_log.unlink(missing_ok=True)

    for province in ("ON", "BC", "YT"):
        pdf = _province_pdf(cfg.raw_pdf_dir, province)
        if pdf is None:
            logger.warning("  [%s] no PDF found; skipping", province)
            continue
        logger.info("  [%s] extracting from %s", province, pdf.name)
        pdf_hashes[province] = _pdf_hash(pdf)
        province_records = asyncio.run(
            vision_extract_province(
                pdf,
                province=province,
                client=client,
                model=cfg.extract_model,
                failure_log=failure_log,
            )
        )
        logger.info("  [%s] %d records extracted", province, len(province_records))
        records.extend(province_records)

    logger.info("Phase 1 done in %.1fs (%d total records, tokens so far: %d)",
                time.monotonic() - phase_t0, len(records),
                _tokens(client.costs.snapshot()))

    # Phase 2: (No-op) vision path is pre-validated by pydantic
    logger.info("Phase 2/5: Validation (vision records are pre-validated by pydantic)")

    # Phase 3: NGS mapping
    logger.info("Phase 3/5: NGS mapping (exact -> semantic+LLM verdict)")
    phase_t0 = time.monotonic()
    ngs_records = []
    for docx_path in sorted(cfg.raw_docx_dir.glob("*.docx")):
        ngs_records.extend(parse_ngs_docx(docx_path))
    logger.info("  loaded %d NGS categories from %s",
                len(ngs_records), cfg.raw_docx_dir)
    records = map_ngs(
        records,
        ngs_records,
        client=client,
        embed_model=cfg.embed_model,
        llm_model=cfg.extract_model,
        dim=cfg.embed_dim,
    )
    mapped = sum(1 for r in records if r.NGS_code)
    logger.info("Phase 3 done in %.1fs (%d/%d codes mapped, tokens so far: %d)",
                time.monotonic() - phase_t0, mapped, len(records),
                _tokens(client.costs.snapshot()))

    # Sort once here. codes.json and embeddings.npz are written with positional
    # correspondence (row i in codes.json == row i in .npz). Any filter or reorder
    # AFTER this point and BEFORE build_embeddings() would silently desync the two.
    records.sort(key=lambda r: (r.province, r.fsc_code))

    # Phase 4: Embeddings
    logger.info("Phase 4/5: Embeddings (%d records -> %s @ %d dim)",
                len(records), cfg.embed_model, cfg.embed_dim)
    phase_t0 = time.monotonic()
    emb_arr, ids = build_embeddings(
        records, client=client, model=cfg.embed_model, dim=cfg.embed_dim
    )
    save_npz(embeddings_path, emb_arr, ids)
    logger.info("Phase 4 done in %.1fs", time.monotonic() - phase_t0)

    # Phase 5: Regression
    logger.info("Phase 5/5: Regression check")
    phase_t0 = time.monotonic()
    previous = _latest_previous(cfg.output_dir, cfg.version)
    if previous is not None:
        logger.info("  comparing against %s", previous.parent.name)
        old_records = [FeeCodeRecord(**d) for d in json.loads(previous.read_text())]
        report = diff(new=records, old=old_records)
        ok, reasons = check(report, golden_set=set(cfg.golden_set))
        (cfg.diagnostics_dir / "regression_diff.txt").write_text(format_report(report))
        if not ok and cfg.accept_regression is None:
            raise RuntimeError("Regression gate failed:\n" + "\n".join(reasons))
    else:
        logger.info("  no prior version found; skipping regression diff")
    logger.info("Phase 5 done in %.1fs", time.monotonic() - phase_t0)

    # Write outputs
    codes_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in records], indent=2),
        encoding="utf-8",
    )

    row_counts = {
        p: sum(1 for r in records if r.province == p) for p in ("ON", "BC", "YT")
    }
    manifest = Manifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        row_counts=row_counts,
        source_pdf_hashes=pdf_hashes,
        models={"embed": cfg.embed_model, "extract": cfg.extract_model},
        regression_override=cfg.accept_regression,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    # Cost diagnostics
    costs = client.costs.snapshot()
    (cfg.diagnostics_dir / "costs.json").write_text(json.dumps(costs, indent=2))
    logger.info(
        "Pipeline complete in %.1fs | rows=%s | tokens=%d | output=%s",
        time.monotonic() - t0, row_counts, _tokens(costs), version_dir,
    )
    for model, s in costs.items():
        logger.info("  %s: %d calls, %d prompt + %d completion tokens",
                    model, s["calls"], s["prompt_tokens"], s["completion_tokens"])

    return PipelineResult(skipped=False, version_dir=version_dir, manifest=manifest)


def _write_jsonl(path: Path, items: list[dict], *, append: bool = False) -> None:
    mode = "a" if append else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _latest_previous(output_dir: Path, current_version: str) -> Path | None:
    candidates = sorted(
        p
        for p in output_dir.glob("v*/codes.json")
        if p.parent.name != f"v{current_version}"
    )
    return candidates[-1] if candidates else None
