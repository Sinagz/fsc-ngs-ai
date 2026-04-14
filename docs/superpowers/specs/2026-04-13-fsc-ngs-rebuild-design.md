# FSC-NGS Rebuild — Umbrella Design + Phase 1 Spec

**Date:** 2026-04-13
**Branch:** `rebuild/pipeline-openai`
**Status:** Draft — pending user review
**Authors:** Pouria Mortezaagha (with Claude)

---

## 1. Summary

The current FSC-NGS cross-province lookup works, but three root-cause issues limit further investment:

1. **Extraction is fragile.** The pipeline switched from a coordinate-based pymupdf parser to Mistral OCR to escape layout brittleness, but OCR over digital-native PDFs loses information the text layer already contains, and silent zero-code windows still require manual re-extraction.
2. **Matching is shallow.** Cosine similarity on BGE-large embeddings over description text misses medical-semantic equivalence across provincial billing dialects, and produces no rationale or confidence per match.
3. **Architecture constrains iteration.** Streamlit singleton + flat JSON + optional GPU embeddings is hard to deploy, hard to version, and exposes no service boundary.

This rebuild replaces the pipeline, the matcher, the UX, and the deployment model across four sequential sub-projects. **Only Phase 1 is spec-grade in this document**; Phases 2-4 are sketched at the contract level so the interfaces between them can be locked now. Each later phase gets its own brainstorm → spec → plan → implement cycle.

---

## 2. Cross-phase contracts (umbrella)

These four contracts apply to every phase. Breaking changes to any of them require a schema version bump and explicit migration.

### Contract A — Canonical `FeeCodeRecord` schema

One pydantic model is the single source of truth for pipeline output, matching input, API responses, and Excel export. Adding or removing a field means bumping `schema_version` and updating every consumer in the same PR. This eliminates the current class of silent-drop bugs where `excel_export.COLUMNS` and `FeeCode.from_dict` can disagree.

The schema is defined in `src/pipeline/schema.py` (section 4.1 below) and imported wherever records are constructed or consumed.

### Contract B — Matching API signature

```python
def search(
    fsc_code: str,
    src: Province,
    dst: Province,
    top_n: int = 5,
    with_rationale: bool = False,
) -> LookupResult: ...
```

This signature is stable across the CLI, Streamlit UI, and (Phase 4) FastAPI service. `with_rationale=False` keeps the fast retrieval-only path; `True` enables the Phase 2 LLM reranker with streamed explanations.

Phase 1 ports the existing cosine-similarity + Jaccard-fallback algorithm into `src/core/matching.py`, reading the new versioned artifacts via `src/core/loader.py`. The old `app/lookup_engine.py` is deleted; `app/main.py` updates its imports. This is a port, not an algorithm change — Phase 1 preserves current matching behaviour so the app keeps working, and Phase 2 replaces the algorithm with OpenAI-embedding retrieval + `gpt-4o` reranking.

### Contract C — Config via `.env` only

Required:
- `OPENAI_API_KEY`

Optional (with defaults):
- `OPENAI_EMBED_MODEL=text-embedding-3-large`
- `OPENAI_EMBED_DIM=1024`
- `OPENAI_EXTRACT_MODEL=gpt-4o-mini`
- `OPENAI_RERANK_MODEL=gpt-4o`  (Phase 2)

No GPU environment variables. No hardcoded keys anywhere. `python-dotenv` loads at CLI entry and at Streamlit startup. `os.environ[...]` is used at the boundary so missing keys fail loudly at startup, not mid-run.

### Contract D — Versioned data artifacts

```
data/parsed/v<YYYY-MM-DD>/
    codes.json           # List[FeeCodeRecord], sorted by (province, fsc_code)
    embeddings.npz       # {"embeddings": (N, D) float32, "record_ids": (N,) int32}
    manifest.json        # row counts, source PDF hashes, model versions, git sha
```

The app loads the newest directory by default (lexically max). Regressions become diffable; rollback is a symlink swap. The `manifest.json` is the authoritative metadata — any runtime code that needs to know "which embed model produced this" reads it from the manifest, never from code.

---

## 3. Phases 2-4 (sketched, not spec-grade)

- **Phase 2 (A — matching quality):** OpenAI embeddings retrieve top-20 candidates, `gpt-4o` reranks to top-N with structured rationale + confidence. Replaces the current cosine-only ranker. Requires no pipeline changes — just a new `src/core/matching.py` implementation.
- **Phase 3 (B — UX):** Streamed rationales, natural-language search ("knee arthroscopy with meniscus repair" → codes), "why not X?" drill-downs. Built on top of the Phase 2 API; no further core changes.
- **Phase 4 (D — deploy):** FastAPI service wrapping `src/core/matching.py`, SQLite or DuckDB replaces the flat JSON load, Streamlit becomes a thin client, Docker + CI.

---

## 4. Phase 1 — Robust pipeline (full spec)

### 4.0 Goal

Produce `codes.json` + `embeddings.npz` from the raw provincial PDFs with verifiable correctness, loud failure modes, no GPU, no Mistral OCR, and <$1 per full pipeline run.

### 4.1 Success criteria (measurable)

1. **Coverage:** ≥99% of the golden set (20 hand-picked codes per province, 60 total) extract with all required fields populated.
2. **Schema compliance:** 100% of output records pass pydantic validation.
3. **Regression detectability:** a PDF update that deletes a chapter fails the pipeline instead of silently dropping ~200 codes.
4. **Reproducibility:** `python -m src.pipeline.run --force` on the same raw PDFs produces byte-identical `codes.json` (modulo timestamps in the manifest).
5. **Cost:** full pipeline run costs <$1 in OpenAI API fees.
6. **No GPU:** `import torch` is removed from the critical path. `requirements.txt` drops `torch` and `sentence-transformers`.

### 4.2 Architecture

New source layout:

```
src/
├── pipeline/
│   ├── __init__.py
│   ├── schema.py            # pydantic FeeCodeRecord, NGSRecord, Manifest
│   ├── io.py                # pymupdf loader, text + layout extraction
│   ├── structural/
│   │   ├── base.py          # StructuralExtractor Protocol + CandidateRow
│   │   ├── ontario.py       # ON-specific fonts, regex, section rules
│   │   ├── bc.py            # BC-specific
│   │   └── yukon.py         # YT-specific
│   ├── semantic.py          # gpt-4o-mini rescue for low-confidence rows
│   ├── ngs_parser.py        # DOCX → NGSRecord (replaces parse_docx_full.py)
│   ├── ngs_mapper.py        # FSC → NGS, gpt-4o-mini assisted
│   ├── embed.py             # OpenAI text-embedding-3-large batch
│   ├── regression.py        # diff vs previous version, fail loud
│   └── run.py               # orchestrator (replaces scripts/run_pipeline.py)
├── core/
│   ├── __init__.py
│   ├── matching.py          # search() — Phase 1 stub, Phase 2 fills this in
│   └── loader.py            # load versioned DB artifacts
├── openai_client.py         # httpx + hishel-cached OpenAI wrapper
└── cli.py                   # typer-based commands
```

**Separations that matter:**

- `src/pipeline/` is the only module tree that writes to `data/parsed/`.
- `src/core/` has no I/O beyond reading the published artifacts — easy to unit-test.
- `src/openai_client.py` is the single chokepoint for OpenAI calls; tests mock this one file.
- `app/` stays small and UI-only; imports from `src/core/`, never from `src/pipeline/`.

**Data flow:**

```
raw/pdf/*.pdf
   │
   ▼  io.load_pdf()                → PageBlock[] (text + fonts + coords)
   ▼  structural.extract()         → CandidateRow + confidence ∈ [0,1]
   ▼  confidence < 0.8 ?
       yes → semantic.rescue()     → CandidateRow (same shape, LLM-resolved)
       no  → pass through
   ▼  validate()                   → FeeCodeRecord (pydantic strict)
   ▼  dedupe(province, fsc_code)   → FeeCodeRecord list per province
   ▼  ngs_mapper.map()             → FeeCodeRecord with NGS fields filled
   ▼  embed.build()                → embeddings.npz (OpenAI, L2-normalized)
   ▼  regression.check()           → pass | FAIL (exit 1 unless --accept-regression)
   ▼  write v<DATE>/{codes, embeddings, manifest}
```

**Orchestration:** `src/pipeline/run.py` replaces `scripts/run_pipeline.py`. Same idempotent behavior (skip steps whose outputs exist, `--force` to redo) but driven by a `Step` dataclass list instead of string-matching on filenames. Each step declares its inputs and outputs so dependency tracking is explicit.

**OpenAI calls are disk-cached** via `hishel` keyed on `(model, prompt_sha256, schema_sha256)`. Re-running the pipeline after fixing a downstream bug doesn't re-spend on OpenAI for steps whose inputs didn't change.

### 4.3 Components

#### 4.3.1 `schema.py` — canonical types

```python
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, ConfigDict

Province = Literal["ON", "BC", "YT"]

class FeeCodeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"] = "1"
    province: Province
    fsc_code: str
    fsc_fn: str
    fsc_description: str
    fsc_chapter: str | None = None
    fsc_section: str | None = None
    fsc_subsection: str | None = None
    fsc_notes: str | None = None
    price: Decimal | None = None
    page: int
    source_pdf_hash: str                          # sha256 of source PDF
    extraction_method: Literal["structural", "semantic"]
    extraction_confidence: float                  # 0..1

    NGS_code: str | None = None                   # None = NOMAP
    NGS_label: str | None = None
    NGS_mapping_method: Literal["exact", "llm", "manual"] | None = None
    NGS_mapping_confidence: float = 0.0
```

`NGSRecord` and `Manifest` follow the same pattern. All are `frozen=True` per the global immutability rule.

#### 4.3.2 `structural/` — per-province extractors

Protocol:

```python
class StructuralExtractor(Protocol):
    PROVINCE: ClassVar[Province]
    CODE_REGEX: ClassVar[re.Pattern[str]]

    def extract(self, pages: list[PageBlock]) -> Iterator[CandidateRow]: ...
```

Each province implementation is ~150-300 lines. Layout rules are declared as module-level constants with provenance docstrings:

```python
# ON MOH Schedule of Benefits, 2024-12 edition.
# Verified against pp. 1, 47, 203. Update tolerances if the publisher
# changes font sizes in the next edition.
CHAPTER_FONT_SIZE = 14.0
SECTION_FONT_SIZE = 11.5
CODE_COLUMN_X = 72.0
CODE_COLUMN_TOLERANCE = 3.0
```

**Confidence scoring is explicit:**

- `1.0` — all fields matched cleanly (regex + font + coord)
- `0.85` — one field had to be recovered from the adjacent line
- `0.6` — row matched the code regex but could not locate price or description
- `0.3` — parser flagged this region as ambiguous (possibly a table header)
- `0.0` — parser explicitly rejects this row

Rows with confidence ≥ 0.8 skip the LLM. Rows with confidence < 0.8 go to the semantic rescue path.

#### 4.3.3 `semantic.py` — LLM rescue

Only sees rows with structural confidence < 0.8. For each row, sends to `gpt-4o-mini`:

- The original text window (5 lines before and after the candidate row).
- The currently detected chapter / section / subsection context.
- A pydantic-derived JSON schema via `response_format={"type": "json_schema", "strict": true}`.

Uses `temperature=0`. Response is validated by the same `FeeCodeRecord` pydantic model — LLM-returned records that fail validation are logged and the row is marked unresolved.

Cached via `hishel` keyed on the prompt hash, so re-running the pipeline after a downstream fix is free. When the LLM returns "cannot extract" the row is written to `data/diagnostics/<DATE>/unresolved.jsonl` and the pipeline continues.

#### 4.3.4 `validate.py` — strict gate

Pydantic does most of the work. On top:

- `fsc_code` matches the province's regex (ON: `[A-Z]\d{3,4}`, BC: `\d{5}`, YT: `\d{4}`).
- `price` is a positive `Decimal` or `None`.
- `page` is within the source PDF's page count.
- `fsc_code` uniqueness within each province (duplicate detection emits both source locations in the error).

Rejects go to `data/diagnostics/<DATE>/validation_rejects.jsonl` with human-readable reasons. The pipeline fails if reject rate exceeds 1% for any province.

#### 4.3.5 `ngs_parser.py`

Parses the two reference DOCX files into `NGSRecord` objects. Structurally similar to today's `parse_docx_full.py` but emits validated pydantic models and writes to `data/parsed/v<DATE>/ngs.json`. No LLM involvement — DOCX structure is clean enough for direct parsing.

#### 4.3.6 `ngs_mapper.py`

Input: validated `FeeCodeRecord` list (NGS fields empty) + `NGSRecord` list.

Process:

1. **Exact match** — if the fee code appears literally in an NGS category's code list, assign with `mapping_method="exact"`, `confidence=1.0`.
2. **Semantic match** — remaining codes get their descriptions embedded via OpenAI, cosine-compared against NGS category descriptions; top-1 with similarity > 0.5 is sent to `gpt-4o-mini` for a yes/no verification with short rationale.
3. **NOMAP** — everything else. Logged to diagnostics.

Cached aggressively — NGS data is static across pipeline runs on the same inputs.

#### 4.3.7 `embed.py`

- Model: `text-embedding-3-large`.
- Dimensions: 1024 (configurable via `OPENAI_EMBED_DIM`). Chosen to match BGE-large's 1024-dim shape so Phase 2 can defer any matrix-shape changes. Can be bumped to 3072 if match quality requires it.
- Batch size: 500. Exponential backoff on rate limits. Total ~7k records → ~15 batches.
- Output `.npz` shape: `{"embeddings": (N, D), "record_ids": (N,)}`, identical to today.
- L2-normalized at write time so cosine similarity is a single dot product at query time.

#### 4.3.8 `regression.py`

Compares new `codes.json` to the most recent committed version. Emits a human-readable diff:

```
Provinces:     ON      BC      YT
Before:        4778    1084    219
After:         4780    1090    219
Added:         +7      +8      +0
Removed:       -5      -2      -0
Field-changed: +12     +3      +1
```

**Fails the pipeline if:**

- Any province's total code count drops by ≥5%.
- Any golden-set code disappears.
- `schema_version` changed without an explicit `--accept-schema-bump` flag.

Override available via `--accept-regression "reason: annual PDF update, manually verified"`. The override reason is stored in `manifest.json`.

#### 4.3.9 `openai_client.py`

Single wrapper around `openai.OpenAI()` with:

- `httpx` transport + `hishel` disk cache (keyed on request body hash).
- Unified retry with exponential backoff on 429 / 5xx.
- Per-request cost accounting logged to `data/diagnostics/<DATE>/costs.json`.
- One injection point for test fakes.

No module in `src/pipeline/` or `src/core/` may import `openai` directly — they go through this wrapper.

### 4.4 Testing, regression, diagnostics

**Test pyramid:**

1. **Unit tests** (`tests/unit/`) — each structural extractor against a tiny synthetic PDF with known contents; pydantic schema round-trips; regex validators for each province's code pattern.
2. **Integration tests** (`tests/integration/`) — full pipeline on a fixture PDF subset (5-10 pages per province); golden set of 20 known codes per province asserted after extraction; NGS mapping against a fixed NGSRecord snapshot.
3. **Snapshot test** (`tests/regression/`) — `codes.json` from the latest commit is the snapshot; new runs diff against it; intentional changes regenerate with `pytest --update-snapshots` and require an explicit commit.
4. **Property tests** (`tests/property/`, hypothesis) — any `FeeCodeRecord` → JSON → `FeeCodeRecord` round-trips unchanged; structural extractors never crash and never emit invalid records on random valid `PageBlock` input.

**Diagnostics artifacts** (written every run to `data/diagnostics/<DATE>/`, gitignored):

- `unresolved.jsonl` — rows the LLM could not extract.
- `validation_rejects.jsonl` — schema-failed rows with reasons.
- `low_confidence.jsonl` — rows with structural confidence < 0.5 for spot audit.
- `regression_diff.txt` — human-readable diff from the prior version.
- `costs.json` — OpenAI token counts per step and dollar estimate.

These are the continuous feedback signal. A human skims them periodically; they are how we learn whether the structural parser is drifting before the golden set breaks.

### 4.5 What is explicitly NOT in Phase 1 (YAGNI)

- **No matching-algorithm improvements.** Phase 1 ports the current cosine + Jaccard logic into `src/core/matching.py` (reading the new versioned artifacts) and deletes `app/lookup_engine.py`. Same algorithm, new location, new data contract. Phase 2 replaces the algorithm.
- **No UI changes.** Streamlit app swap-in is limited to import-path updates (`from app.lookup_engine` → `from src.core.matching`) and a sidebar caption noting the new embedding source.
- **No FastAPI, no DuckDB, no Docker.** Phase 4.
- **No rationale generation in match results.** Phase 2.
- **No new provinces.** Schema supports them via the `Province` Literal union; adding AB or QC is a pure extractor addition in a later project.
- **No NGS category re-derivation.** Continue parsing the existing DOCX files.
- **No batch / CSV lookup.** Future.
- **No OpenAI streaming.** Only Phase 3 UX streams.
- **No auth, rate limiting, or multi-tenancy.** Phase 4.

---

## 5. Dependency changes

### Added

- `pydantic>=2.0` (already present, keep)
- `openai>=1.30`
- `httpx>=0.27`
- `hishel>=0.0.30`
- `typer>=0.12`
- `pdfplumber>=0.11` (complements pymupdf for table-like regions)
- `hypothesis>=6.0` (dev)
- `pytest-cov>=5.0` (dev)

### Removed

- `torch` (no GPU dependency)
- `sentence-transformers` (replaced by OpenAI embeddings)
- `mistralai` (Mistral OCR removed)

### Kept

- `pymupdf`, `pymupdf4llm`, `python-docx`, `rapidfuzz`, `streamlit`, `openpyxl`, `pandas`, `numpy`, `python-dotenv`

---

## 6. Success evidence (what "Phase 1 done" looks like)

1. `python -m src.pipeline.run --force` completes in <10 minutes on a laptop with no GPU, costs <$1.
2. `pytest` passes, including the 60-code golden set and the snapshot test.
3. The Streamlit app launches and returns identical or improved results for a hand-picked set of anchor codes compared against the pre-rebuild version.
4. `requirements.txt` contains no `torch`, no `sentence-transformers`, no `mistralai`.
5. A deliberate PDF-corrupting test (delete a chapter from a copy of one PDF) causes the regression gate to fail loudly.
6. `docs/` has an updated README and an updated `CLAUDE.md` reflecting the new pipeline.

---

## 7. Open questions / deferred decisions

- **Golden set curation** — who picks the 60 codes, and based on what criteria (one per chapter? highest-billed? hardest-to-extract)? Deferred to plan phase; can be mechanical (random stratified sample) if no domain input is available.
- **Embedding dimension at rollout** — start at 1024 to minimize Phase 2 churn; revisit in Phase 2 if reranker quality is bottlenecked by retrieval recall.
- **Diagnostics retention** — currently gitignored. If we want historical drift analysis, we may later add a `data/diagnostics/archive/` that keeps the last N runs. Not in scope for Phase 1.
- **CI runtime** — snapshot tests require committing `codes.json`; full extraction-in-CI would need the raw PDFs which are not redistributable. Resolution: CI runs only unit + property tests and the snapshot diff; full integration runs locally before each merge.
