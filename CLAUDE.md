# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A Streamlit app that maps physician **Fee Service Codes (FSC)** across Canadian provinces (ON, BC, YT) and links them to **CIHI National Grouping System (NGS)** categories. A user enters one FSC code from one province and the app returns the closest equivalents in the other provinces.

The project has two distinct phases:

1. **Offline data pipeline** (`src/pipeline/` + `python -m src.cli run`) — turns raw PDFs/DOCXs into versioned `data/parsed/v<DATE>/{codes.json, embeddings.npz, manifest.json}` artifacts. Run rarely; requires source files in `data/raw/` and an `OPENAI_API_KEY`.
2. **Runtime app** (`app/` + `src/core/`) — loads the newest versioned artifacts and serves lookups. Run often, no PDF/DOCX dependency.

**Preferred model for this repo: Opus.** Pipeline parsers and the lookup engine both reward deeper reasoning over speed, so default to Opus rather than the Haiku/Sonnet tiers suggested globally.

**Phase 1 design + plan:**
- `docs/superpowers/specs/2026-04-13-fsc-ngs-rebuild-design.md`
- `docs/superpowers/plans/2026-04-13-fsc-ngs-phase1-pipeline.md`

## Common commands

```bash
# Install (Windows + Git Bash; pip, not uv — requirements.txt is the authoritative manifest)
pip install -r requirements.txt

# Configure OpenAI key
cp .env.example .env       # then set OPENAI_API_KEY=sk-...

# Run the full pipeline (defaults to today's date as the version tag)
python -m src.cli run                        # skip-if-output-exists
python -m src.cli run --force                # redo even if outputs exist
python -m src.cli run --version 2026-04-14   # explicit version tag
python -m src.cli run --accept-regression "annual PDF update, manually verified"

# Tests
pytest                       # unit + integration + property + regression
pytest -m unit               # fast, no I/O
pytest -m integration        # full pipeline against fixtures or real artifacts
pytest -m regression         # snapshot diff

# Launch the app
streamlit run app/main.py    # http://localhost:8501
```

### Starting from a fresh clone

The committed pipeline artifacts live under `data/parsed/v*/codes.json` and `manifest.json` (one directory per version; `embeddings.npz` is gitignored). To rebuild from source, populate `data/raw/` (gitignored, must be obtained manually from the provincial publishers):

```
data/raw/pdf/
  moh-schedule-benefit-*.pdf          # Ontario MOH Schedule of Benefits
  msc_payment_schedule_*.pdf          # BC MSC Payment Schedule
  yukon_physician_fee_guide_*.pdf     # Yukon Physician Fee Guide
data/raw/docx/
  Fee Codes and Grouping Rules.docx           # CIHI NGS grouping rules
  NPDB National Grouping System Categories.docx
```

Glob patterns are declared in `_province_pdf()` inside `src/pipeline/run.py` — update those if the publishers change the naming scheme.

## Architecture

### Data flow (pipeline)

```
data/raw/pdf/*.pdf             data/raw/docx/*.docx
        │                              │
        ▼ src/pipeline/io.load_pdf     ▼ src/pipeline/ngs_parser
   PageBlock list                  NGSRecord list
        │                              │
        ▼ structural/{ontario,bc,yukon} extractors
   CandidateRow + confidence ∈ [0,1]
        │
        ▼ semantic.rescue (gpt-5.4-mini, only for confidence < 0.8)
   CandidateRow (origin="structural" or "semantic")
        │
        ▼ validate (province regex, dedupe, positive price)
   FeeCodeRecord list ───────────────┐
                                     ▼ ngs_mapper (exact → semantic+LLM verdict → NOMAP)
                              FeeCodeRecord with NGS fields filled
                                     │
                                     ▼ embed (OpenAI text-embedding-3-large, L2-normalized)
                              embeddings.npz
                                     │
                                     ▼ regression (diff vs prior version, fail loud)
                              data/parsed/v<DATE>/{codes.json, embeddings.npz, manifest.json}
```

The pipeline is **idempotent and resumable** — `run_pipeline()` skips when the output `codes.json` and `manifest.json` already exist for the target version. Pass `--force` to redo. OpenAI calls are cached on disk via hishel, so repeated runs are free even when not skipped.

### Runtime (app)

- `app/main.py` — Streamlit UI. Uses `@st.cache_resource` to load the engine once per process.
- `src/core/loader.py` — `load_latest()` reads the newest `data/parsed/v*/` bundle: records, embeddings, record_ids, manifest.
- `src/core/matching.py` — `search()` is the canonical lookup API (Contract B in the design spec). Two paths:
  - **Semantic (preferred)** — cosine over L2-normalised OpenAI embeddings (loaded from `embeddings.npz`).
  - **Jaccard fallback** — blended `0.4 * jaccard + 0.6 * overlap` on tokenised descriptions, used when `embeddings.npz` is missing.
- `app/excel_export.py` — openpyxl workbook generator. Column list is **derived from `FeeCodeRecord.model_fields`** so adding a schema field automatically appears in the Excel export (no parallel `COLUMNS` list to drift).

Both paths rank purely by description similarity. **Same-NGS is flagged but does not boost scores** — two provinces can legitimately assign equivalent procedures to different NGS codes. Do not add NGS-based ranking boosts without explicit discussion (see the comment block around `_ngs_match()` in `src/core/matching.py`).

### Embeddings

- Model: OpenAI `text-embedding-3-large`, default 1024-dim (configurable via `OPENAI_EMBED_DIM`).
- No GPU required. No `torch`, no `sentence-transformers`.
- Cost: ~$0.15 per full pipeline run (7000+ records × ~200 tokens each).
- Regenerate any time `codes.json` changes — anchor→embedding row indices are positional. `src/pipeline/run.py` enforces this by sorting records once before both `build_embeddings` and `codes.json` are written; do NOT add filters between the sort and `build_embeddings()`.

### Record schema

Canonical types live in `src/pipeline/schema.py`:
- `FeeCodeRecord` — frozen pydantic, `extra="forbid"`. The single source of truth across pipeline output, matching input, API responses, and Excel export.
- `CandidateRow` — pre-validation record with `confidence` and `origin: Literal["structural", "semantic"]`.
- `NGSRecord` — `code_refs: tuple[str, ...]` (immutable inside frozen model).
- `Manifest` — frozen, written per pipeline run.
- `PageBlock` — pymupdf text span with layout, `extra="forbid"`.

Adding a field means bumping `schema_version` (currently `Literal["1"]`) and updating consumers in the same PR.

### OpenAI client

`src/openai_client.py` is the single chokepoint. No module under `src/pipeline/` or `src/core/` imports `openai` directly. Features:
- httpx transport wrapped by hishel `SyncSqliteStorage` + `SyncCacheTransport(policy=FilterPolicy(use_body_key=True))` — caches POST responses keyed on request body so re-runs of the pipeline don't re-spend.
- `chat_json(prompt, schema, ...)` — strict JSON-schema response_format with pydantic validation, with retry + fast-fail on deterministic errors.
- `embed(texts, model, dim, batch_size=500)` — batched with retry, returns vectors in the input order (sorted by `Embedding.index`).
- `CostTracker` — per-model token counts; `client.costs.snapshot()` is dumped to `data/diagnostics/<DATE>/costs.json` at the end of each run.

## Conventions and gotchas

- **Committed vs generated data.** The committed pipeline artifacts are `data/parsed/v<DATE>/codes.json` and `manifest.json`. `data/raw/`, `data/diagnostics/`, and `data/parsed/v*/embeddings.npz` are gitignored and regenerated locally. Never commit raw PDFs/DOCXs.
- **Paths are resolved from repo root** via `ROOT = Path(__file__).resolve().parent.parent` in `app/main.py` and `src/cli.py`. Use the same pattern when adding new entry points.
- **Province list** is `Literal["ON", "BC", "YT"]` in `src/pipeline/schema.py`. Adding a province means: extending the Literal, adding a new structural extractor that subclasses `OntarioExtractor` (or `BCExtractor` for digit-only codes), adding a glob pattern to `_province_pdf()` in `run.py`, adding the regex to `PROVINCE_REGEX` in `validate.py`, and adding a column to `regression.py`'s `PROVINCES` list.
- **PDF layout parsing depends on font sizes and column x-coordinates.** Per-province extractors declare them as `ClassVar` constants with provenance comments naming the verified PDF edition. If the source PDFs are updated, those values may need re-tuning; the regression gate will fail loudly on a >=5% code-count drop.
- **Streamlit singleton.** `@st.cache_resource` keeps the loaded records + embeddings alive across reruns. If you change loading semantics, ensure cache invalidation works.
- **`.env` is required** for the pipeline (`OPENAI_API_KEY`). The Streamlit app does NOT call OpenAI in Phase 1 — it only reads the pre-built artifacts.
- **Windows + Git Bash.** Use forward slashes in paths when invoking scripts. Some tools need `winpty` for interactive sessions.
- **Sort-then-embed coupling.** `src/pipeline/run.py` sorts records by `(province, fsc_code)` before writing both `codes.json` and `embeddings.npz`. Row `i` in `codes.json` corresponds to row `i` in `.npz`. Any filter or reorder between the sort and `build_embeddings()` would silently desync the two artifacts. The orchestrator carries an inline comment about this; preserve it on edits.
