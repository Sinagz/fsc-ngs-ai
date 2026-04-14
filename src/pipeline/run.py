"""Pipeline orchestrator. Idempotent (skip-if-output-exists), --force to redo."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.openai_client import OpenAIClient
from src.pipeline.embed import build_embeddings, save_npz
from src.pipeline.io import load_pdf, pdf_hash
from src.pipeline.ngs_mapper import map_ngs
from src.pipeline.ngs_parser import parse_ngs_docx
from src.pipeline.regression import check, diff, format_report
from src.pipeline.schema import FeeCodeRecord, Manifest
from src.pipeline.semantic import rescue
from src.pipeline.structural.bc import BCExtractor
from src.pipeline.structural.ontario import OntarioExtractor
from src.pipeline.structural.yukon import YukonExtractor
from src.pipeline.validate import validate

PROVINCE_EXTRACTORS = {
    "ON": OntarioExtractor(),
    "BC": BCExtractor(),
    "YT": YukonExtractor(),
}


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
    extract_model: str = "gpt-4o-mini"
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


def run_pipeline(
    cfg: PipelineConfig, *, client: OpenAIClient | None
) -> PipelineResult:
    version_dir = cfg.output_dir / f"v{cfg.version}"
    codes_path = version_dir / "codes.json"
    embeddings_path = version_dir / "embeddings.npz"
    manifest_path = version_dir / "manifest.json"

    if not cfg.force and codes_path.exists() and manifest_path.exists():
        return PipelineResult(skipped=True, version_dir=version_dir, manifest=None)

    if client is None:
        raise RuntimeError("OpenAIClient required when pipeline is not skipping")

    version_dir.mkdir(parents=True, exist_ok=True)
    cfg.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Clear diagnostics from any prior run so a --force re-run doesn't silently
    # concatenate this run's output with stale rows.
    for name in ("unresolved.jsonl", "validation_rejects.jsonl"):
        (cfg.diagnostics_dir / name).unlink(missing_ok=True)

    # Extract + rescue per province
    all_candidates = []
    pdf_hashes: dict[str, str] = {}
    context_lines: dict[tuple[int, str], str] = {}
    for province, extractor in PROVINCE_EXTRACTORS.items():
        pdf = _province_pdf(cfg.raw_pdf_dir, province)
        if pdf is None:
            continue
        h = pdf_hash(pdf)
        pdf_hashes[province] = h
        pages = load_pdf(pdf)
        rows = list(extractor.extract(pages, source_pdf_hash=h))
        # For now, context = empty; rescue can be enriched later
        rescued, unresolved = rescue(
            rows,
            client=client,
            model=cfg.extract_model,
            context_lines=context_lines,
            threshold=0.8,
        )
        _write_jsonl(
            cfg.diagnostics_dir / "unresolved.jsonl",
            [r.model_dump(mode="json") for r in unresolved],
            append=True,
        )
        all_candidates.extend(rescued)

    # Validate
    records, rejects = validate(all_candidates)
    _write_jsonl(
        cfg.diagnostics_dir / "validation_rejects.jsonl",
        [
            {"row": e.candidate.model_dump(mode="json"), "reason": e.reason}
            for e in rejects
        ],
    )

    # NGS mapping
    ngs_records = []
    for docx_path in sorted(cfg.raw_docx_dir.glob("*.docx")):
        ngs_records.extend(parse_ngs_docx(docx_path))
    records = map_ngs(
        records,
        ngs_records,
        client=client,
        embed_model=cfg.embed_model,
        llm_model=cfg.extract_model,
        dim=cfg.embed_dim,
    )
    # Sort once here. codes.json and embeddings.npz are written with positional
    # correspondence (row i in codes.json == row i in .npz). Any filter or reorder
    # AFTER this point and BEFORE build_embeddings() would silently desync the two.
    records.sort(key=lambda r: (r.province, r.fsc_code))

    # Embeddings
    emb_arr, ids = build_embeddings(
        records, client=client, model=cfg.embed_model, dim=cfg.embed_dim
    )
    save_npz(embeddings_path, emb_arr, ids)

    # Regression
    previous = _latest_previous(cfg.output_dir, cfg.version)
    if previous is not None:
        old_records = [FeeCodeRecord(**d) for d in json.loads(previous.read_text())]
        report = diff(new=records, old=old_records)
        ok, reasons = check(report, golden_set=set(cfg.golden_set))
        (cfg.diagnostics_dir / "regression_diff.txt").write_text(format_report(report))
        if not ok and cfg.accept_regression is None:
            raise RuntimeError("Regression gate failed:\n" + "\n".join(reasons))

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
    (cfg.diagnostics_dir / "costs.json").write_text(
        json.dumps(client.costs.snapshot(), indent=2)
    )

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
