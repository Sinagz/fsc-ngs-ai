# LLM-Vision PDF Extraction — Design Spec

**Date:** 2026-04-14
**Branch (proposed):** `rebuild/pipeline-vision` (branch off `main`, not off `rebuild/pipeline-openai`)
**Status:** Draft — pending user review
**Authors:** Pouria Mortezaagha (with Claude)
**Supersedes (partially):** `docs/superpowers/specs/2026-04-13-fsc-ngs-rebuild-design.md` §4 (Phase 1 pipeline), specifically the structural-then-semantic extraction path. The umbrella contracts (A–D) and Phases 2–4 are unchanged.

---

## 1. Summary

Replace the province-specific structural PDF parsers and the confidence-gated semantic rescue with a single vision-based LLM extraction pass. One rendered page image per pass, with one-page of prior context attached, processed by `gpt-5.4-mini` under a deterministic "emit-where-it-completes" prompt rule. Downstream `validate → ngs_mapper → sort → embed → regression` is unchanged. Schema bumps `v1 → v2`; `ExtractionMethod` becomes `Literal["vision"]`. The structural and semantic modules are deleted; dead code is not kept around.

## 2. Motivation

Three problems with the structural-first pipeline, in order of severity:

1. **Silent field-level corruption.** Inspection of committed artifact `data/parsed/v2026-04-14/codes.json` reveals all 1,251 BC records carry single-digit `price` values (`"7"`, `"5"`, `"2"`) where the source document lists multi-dollar amounts like `83.68`. The BC structural extractor is pulling the wrong column. The regression gate is row-count-based and never fires. Nobody noticed because the app ranks on description similarity and hides price in the UI.

2. **Per-edition brittleness.** Recent commits: `b89090c feat: rewrite Ontario extractor for 2025-03-19 PDF edition`, `2f89fa0 fix: Yukon extractor body font size 10.0 -> 9.0`. Each annual PDF update forces a manual retune of font sizes and column x-coordinates. Ontario alone is 363 lines of province-specific structural code.

3. **Encoding corruption.** Ontario records contain `�` (U+FFFD) where `½` should be — a pymupdf text-extraction edge case. Vision bypasses the text layer entirely.

The vision approach trades ~$25 per run for elimination of all three classes.

## 3. Decisions locked during brainstorming

| # | Question | Decision |
|---|----------|----------|
| Q1 | Scope of replacement | **B.** PDF → `FeeCodeRecord` in one LLM pass. `validate`, `ngs_mapper`, `embed`, `regression` downstream stay. |
| Q2 | Input format | **C.** Rendered page images via vision. |
| Q3 | Sliding-window shape | **B.** Pairs, step 1, overlap-aware "emit-where-it-completes" prompt. Defensive dedup on `(province, fsc_code)` as insurance. |
| Q4 | Chapter/section source | **A.** `pymupdf.get_toc()` for ON/BC (272 + 620 entries). Inline-only for YT (0 entries, nulls acceptable). |
| Q5 | Model | **A.** `gpt-5.4-mini` with a knob to escalate. |
| Q6 | Old-code disposition | **A.** Full delete of `src/pipeline/structural/`, `semantic.py`, `io.py`. No coexistence. |
| — | Response typing | Pydantic `response_format`, not dict parsing. Consistent with existing `openai_client.chat_json()`. |

## 4. Architecture & data flow

```
data/raw/pdf/<province>.pdf            data/raw/docx/*.docx
         │                                     │
         ▼                                     ▼
 render_pages(dpi=144, colorspace=RGB, fmt=PNG)   ngs_parser (unchanged)
   yields (page_idx, png_bytes)                   → NGSRecord list
         │
         ▼
 build_section_map(pymupdf.get_toc())
   → {page_idx: SectionContext}  # {} for YT
         │
         ▼
 build_windows(num_pages, section_map)
   yields Window(target_page, context_page|None, section_hints)
         │
         ▼
 extract_window(window, images) ── openai_client.chat_vision_json
   prompt: emit-where-it-completes, TOC-injected section hints
   response_format: WindowExtraction(records: list[VisionRecord])
         │
         ▼
 merge_windows  (defensive dedup on (province, fsc_code))
   → list[VisionRecord]
         │
         ▼
 to_fee_code_record(province, source_pdf_hash, extraction_method="vision")
         │
         ▼
 validate (unchanged — province regex, positive price, dedupe)
         │
         ▼
 ngs_mapper (unchanged: exact → semantic+LLM verdict → NOMAP)
         │
         ▼
 sort by (province, fsc_code)   ← preserves sort-then-embed coupling
         │
         ▼
 embed (unchanged)  +  write codes.json, embeddings.npz, manifest.json
         │
         ▼
 regression (row-count gate + new per-field spot-check on golden set)
```

**Three load-bearing invariants:**
- `src/openai_client.py` remains the sole OpenAI chokepoint. `chat_vision_json()` is a thin addition, same hishel cache, same `CostTracker`, same fast-fail on deterministic errors.
- `validate → ngs_mapper → sort → embed → regression` is byte-identical to today. Only the code producing the `FeeCodeRecord` list is new.
- Sort-then-embed coupling (row `i` of `codes.json` ↔ row `i` of `embeddings.npz`) is preserved. The existing inline comment in `run.py` stays.

## 5. Window construction & prompt contract

### 5.1 Windows

For a PDF with `N` pages (1-indexed as printed):
- Window 1: target page 1, context = [page 1 alone] *(seed case)*.
- Window k (2 ≤ k ≤ N): target page k, context = [page k-1, page k].

**N windows total. Every page is the target of exactly one window.** Every multi-page PDF entry is fully visible within its owning window.

### 5.2 Emit rule (verbatim in prompt)

> Emit an entry **iff its final visible line appears on the TARGET PAGE**. If an entry on the target page is cut off at the bottom, omit it — the next window owns it. If an entry on the non-target (earlier) page is already complete before the target page, omit it — a prior window already owned it.

Each entry has exactly one owning window. Defensive dedup on `(province, fsc_code)` — keep highest `extraction_confidence`, tiebreak on lowest `page` — remains as insurance against model slip.

### 5.3 Prompt spine

```
SYSTEM:
You are extracting fee-code entries from the {PROVINCE} physician fee schedule.
One or two pages are shown. The TARGET PAGE is page {P}.

Emit entries iff their final visible line appears on page {P}.
Skip cut-off entries at the bottom of the TARGET PAGE — a later window owns them.
Skip entries whose end lies on the earlier page — a prior window already owned them.

SECTION CONTEXT FOR PAGE {P}  (authoritative — use these exact strings when set):
  chapter:    {TOC L1 | null}
  section:    {TOC L2 | null}
  subsection: {TOC L3 | null}
If a field is null, read any header visible on the TARGET PAGE (Yukon only).

FIELD RULES:
  fsc_code           : code exactly as printed.
  fsc_fn             : short function/name string.
  fsc_description    : full description as one string; join wrapped lines with a space.
  fsc_notes          : rich notes paragraphs after the code, or null.
  price              : dollar amount for this code (strip the $). Null if no price shown.
  page               : the TARGET PAGE number, 1-indexed.
  extraction_confidence : your calibrated 0.0–1.0 confidence for this record.

USER: [image of page P-1]   (omitted for window 1)
USER: [image of page P, labeled "TARGET PAGE"]
```

### 5.4 Response schema

```python
# src/pipeline/vision/schema.py
class VisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    extraction_confidence: float = Field(ge=0.0, le=1.0)

class WindowExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[VisionRecord]
```

`src/openai_client.chat_vision_json()` validates the response against `WindowExtraction`, returning typed pydantic instances. The orchestrator then promotes each `VisionRecord` to `FeeCodeRecord` by attaching `schema_version="2"`, `source_pdf_hash`, `extraction_method="vision"`; NGS fields remain null for `ngs_mapper` to populate.

## 6. Schema bump & migration

### 6.1 `src/pipeline/schema.py` diff

```python
# Before
schema_version: Literal["1"] = "1"
ExtractionMethod = Literal["structural", "semantic"]

# After
schema_version: Literal["2"] = "2"
ExtractionMethod = Literal["vision"]
```

`CandidateRow` and `PageBlock` are deleted — no consumers after the cutover. `FeeCodeRecord`, `NGSRecord`, `Manifest` retain their shape.

### 6.2 New modules (all under `src/pipeline/vision/`)

| File | Responsibility |
|------|----------------|
| `render.py` | `render_pages(pdf_path, *, dpi=144) -> Iterator[tuple[int, bytes]]` using `page.get_pixmap(dpi=144, colorspace=pymupdf.csRGB)` → `pix.tobytes("png")`. DPI, colorspace, and format are module-level constants — changing them invalidates every hishel cache entry. |
| `toc.py` | `build_section_map(pdf_path) -> dict[int, SectionContext]`. Walks `get_toc()` forward-fill; empty dict when no outline. |
| `windows.py` | `build_windows(num_pages, section_map) -> Iterator[Window]`. N windows, target page labelled. |
| `extract.py` | `async extract_window(window, images) -> list[VisionRecord]`. One `chat_vision_json` call per window; retry-once-on-pydantic-failure logic. Uses the module-level `openai_client` singleton. |
| `orchestrate.py` | `async extract_province(pdf_path, province, *, concurrency=20) -> list[FeeCodeRecord]`. Bounded fan-out via `asyncio.Semaphore`, merge + dedup, emit typed records. |

### 6.3 `src/openai_client.py` addition

Mirrors the existing `chat_json` signature shape (keyword-only args, deterministic-error fast-fail, exponential backoff on transport/API errors), adds `images: list[bytes]` and returns via `asyncio`:

```python
async def chat_vision_json(
    self,
    *,
    prompt: str,
    images: list[bytes],       # PNG bytes, labelled in the user-message order
    schema: type[T],
    model: str = "gpt-5.4-mini",
    system: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> T: ...
```

Same hishel storage, same `CostTracker.add(...)`, same fast-fail on deterministic errors (`json.JSONDecodeError`, `pydantic.ValidationError`). The request body includes base64 image payloads plus the prompt, so hishel's body-keyed cache invalidates/reuses automatically.

### 6.4 Deletions (one reviewable commit)

- `src/pipeline/structural/` — all four files
- `src/pipeline/semantic.py`
- `src/pipeline/io.py`
- `tests/pipeline/test_structural_*.py`, `tests/pipeline/test_semantic_*.py`
- `run.py`: strip the "for each province, call structural then semantic rescue" block; replace with `vision.orchestrate.extract_province(...)`

### 6.5 `run.py` rewire

The only change above the sort step:

```python
# Before
candidates = []
for province, pdf_path in _province_pdfs().items():
    structural = extract_structural(province, pdf_path)
    rescued = semantic_rescue([c for c in structural if c.confidence < 0.8])
    candidates.extend(structural + rescued)
records = validate(candidates)

# After
records: list[FeeCodeRecord] = []
for province, pdf_path in _province_pdfs().items():
    records.extend(await vision.extract_province(pdf_path, province))
records = validate(records)
```

Everything from `validate(records)` downward is unchanged.

### 6.6 Artifact migration

Because `schema_version` bumps to `"2"`, `src/core/loader.py` must refuse v1 artifacts and raise a clear error directing the user to rerun the pipeline. The current v1 bundle at `data/parsed/v2026-04-14/` stays in git as a historical snapshot. First v2 run produces `data/parsed/v<NEW-DATE>/` alongside it; the Streamlit loader picks the newest by directory name sort. No in-place overwrite.

## 7. Cost, concurrency, caching

### 7.1 Expected full-run cost (`gpt-5.4-mini` vision, 1,644 pages)

| Bucket | Calculation | ~Cost |
|--------|-------------|-------|
| Input (vision + prompt) | 1,644 × ~2.5k tokens × $2.50/1M | ~$10 |
| Output (JSON records) | 1,644 × ~800 tokens × $10/1M | ~$13 |
| NGS mapping + embeddings | unchanged | ~$2 |
| **Total** | | **~$25** |

Re-runs with unchanged PDFs are free — image bytes are byte-identical → every vision call is a hishel cache hit.

### 7.2 Rendering contract (pinned)

```python
# src/pipeline/vision/render.py
PAGE_DPI = 144
COLORSPACE = pymupdf.csRGB
IMAGE_FORMAT = "png"
# DO NOT EDIT WITHOUT BUMPING schema_version — these values are cache-key inputs.
```

144 DPI keeps 9pt body text legible in OpenAI's "high-detail" image tier. RGB preserves blue hyperlink cues. PNG is lossless and deterministic (JPEG is not).

### 7.3 Concurrency

`asyncio.Semaphore(20)` bounds concurrent `chat_vision_json` calls.
- Serial wall-clock ≈ 4–5 h.
- At 20× concurrency ≈ 15–25 min.
- `gpt-5.4-mini` tier-3 limits (5k RPM, 2M TPM) comfortably absorb 20× (peak ~250k TPM).

Observability via `tqdm_asyncio` progress bar over windows.

### 7.4 Resumability (three layers)

1. **Pipeline-level:** `run_pipeline()` skips when `codes.json + manifest.json` exist for the target version (existing behavior).
2. **Window-level:** hishel cache makes any mid-run crash resume for free on the next invocation.
3. **Failure isolation:** a single window exception is caught, logged to `data/diagnostics/<DATE>/window_failures.jsonl`, contributes zero records; the regression gate catches cumulative damage.

### 7.5 Diagnostics per run (`data/diagnostics/<DATE>/`)

- `costs.json` — `CostTracker.snapshot()` (existing).
- `window_failures.jsonl` — `{province, page, error_class, message}` per failed window.
- `confidence_histogram.json` — `extraction_confidence` distribution per province; surfaces quality drift across editions.
- `low_confidence.jsonl` — records where `extraction_confidence < 0.5`, not dropped, flagged for inspection.

## 8. Validation, error handling, testing

### 8.1 Error-handling matrix

| Failure | Layer | Response |
|---------|-------|----------|
| Transient network / rate limit | `openai_client` retry | 3× exponential backoff |
| Deterministic API error | `openai_client` fast-fail | Raise, abort run |
| Pydantic validation on response | `extract.extract_window` (not `chat_vision_json`) | Catch `ValidationError`, retry once at `temperature=0.2`; on second failure emit zero records + log to `window_failures.jsonl`. Keeps `openai_client`'s deterministic-error fast-fail convention intact — retry policy lives in the caller. |
| `fsc_code` fails province regex | `validate.py` | Drop row, log |
| Duplicate `(province, fsc_code)` | `orchestrate.merge_windows` | Keep highest `extraction_confidence`, tiebreak lowest `page` |
| Window exception (any other) | `orchestrate` | Catch → `window_failures.jsonl` |
| PDF unreadable / 0 pages | `render.py` | Fail loud, abort run |

### 8.2 Regression gates

1. **Row-count gate (existing).** Per-province count must be ≥95% of prior version; smaller drops require `--accept-regression "<reason>"`.
2. **New: per-field spot-check gate.** For every code in `tests/fixtures/golden_codes.json` (60-code set, 20 per province), assert that `fsc_code`, `fsc_description`, and `price` have not changed since the prior run. Changes require the same `--accept-regression` override with explicit reason. This is the defence against silent field-level corruption (BC-price-bug class).

### 8.3 Test plan (replaces deleted structural/semantic tests)

| Category | File | Purpose |
|----------|------|---------|
| Unit | `tests/pipeline/vision/test_windows.py` | N pages → N windows; target-page labels; 1-page PDF edge case |
| Unit | `tests/pipeline/vision/test_toc.py` | TOC forward-fill; YT returns empty map |
| Unit | `tests/pipeline/vision/test_render.py` | `get_pixmap(144 DPI, RGB, PNG)` is byte-identical across runs — cache-key invariant |
| Unit | `tests/pipeline/vision/test_orchestrate.py` | Dedup rules; confidence tiebreaks |
| Property | `tests/pipeline/vision/test_properties.py` | `build_windows(N)` count ≡ N; no dup `(province, fsc_code)`; `price` ≥ 0 or None |
| Integration | `tests/pipeline/test_pipeline_integration.py` | 10-page PDF slice + recorded hishel cache → byte-identical `codes.json` fixture |
| Regression | `tests/pipeline/test_regression.py` | Row-count gate + golden-set field spot-check gate |
| Schema | `tests/pipeline/test_schema_v2.py` | Loader rejects v1; v2 round-trips cleanly |

Integration tests pre-populate `tests/fixtures/hishel_cache.sqlite` with recorded responses so CI runs offline without spending tokens or requiring `OPENAI_API_KEY`. Any drift in DPI, PNG format, or prompt text invalidates the cache and surfaces as integration-test failures — exactly the signal we want.

### 8.4 Manual verification (blocks cutover PR merge)

After the first successful full run on `rebuild/pipeline-vision`:

- Spot-check 20 random records per province against source PDFs.
- Focus on BC prices specifically — confirm the silent-corruption bug is dead.
- Attach verification notes to the PR.

This is the judgment call automated gates cannot make.

## 9. Out of scope

- NGS mapping algorithm changes (`src/pipeline/ngs_mapper.py` unchanged).
- Embedding model swap (`text-embedding-3-large` stays, per umbrella Contract C).
- Matching algorithm changes (Phase 2 concern).
- Streamlit UI changes.
- Adding a fourth province.
- Any changes to `ngs_parser.py` or DOCX handling.

## 10. Rollout checklist

- [ ] Create branch `rebuild/pipeline-vision` from `main`.
- [ ] Land schema v2 + `chat_vision_json` addition in one commit.
- [ ] Land `src/pipeline/vision/` modules + rewired `run.py` in one commit.
- [ ] Land deletion of structural + semantic + tests in one commit.
- [ ] Land new vision tests + regression gate upgrade in one commit.
- [ ] Run full pipeline end-to-end on real PDFs; confirm ~$25 cost and 15–25 min wall-clock.
- [ ] Compare v2 artifact against committed v1 on golden set; sanity-check BC prices.
- [ ] Manual 20-record-per-province spot-check attached to PR.
- [ ] Merge PR, delete old branch `rebuild/pipeline-openai` if no longer needed.

## 11. Open items (tracked for implementation plan)

- DPI sanity-check — before cutover, render one letter-size page at 144 DPI and confirm OpenAI reports it as "high-detail" tier in the usage block. If tier thresholds have shifted, drop to 120 DPI (same tier, smaller body text).
- Temperature — `chat_vision_json` defaults to `0.0`; retry on pydantic-validation failure uses `0.2` exactly once, then gives up on the window.
- Prompt caching — the SYSTEM portion of the prompt is stable across windows; structuring it for OpenAI prompt-cache-hits would cut input cost further. Investigate during implementation.
- YT chapter coverage — if inline-only gives empty chapters for most YT records, consider the header-scan pass (Q4 option C) as a follow-up, not a blocker.
- `openai_client` async story — existing `chat_json` is sync and stays sync (used by `ngs_mapper`). `chat_vision_json` is net-new and async from the start. No conversion of existing callers needed.
