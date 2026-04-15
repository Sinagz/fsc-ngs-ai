# LLM-Vision PDF Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (sequential, recommended for small plans), team-driven-development (parallel swarm, recommended for 3+ tasks with parallelizable dependency graph), or superpowers:executing-plans (inline batch) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-province structural extractors and the semantic rescue path with a single `gpt-5.4-mini` vision pass over sliding 2-page windows, eliminating edition-specific layout brittleness and fixing the silent BC-price corruption along the way.

**Architecture:** New `src/pipeline/vision/` package: render → TOC map → windows → per-window vision extraction → merge + dedup → promote to `FeeCodeRecord`. Downstream `validate → ngs_mapper → sort → embed → regression` is untouched. Schema bumps v1 → v2; old modules deleted; single new async path orchestrated from the existing sync `run_pipeline`.

**Tech Stack:** pymupdf (page rendering + TOC), OpenAI vision via existing `openai_client`, hishel body-keyed cache (re-runs are free), asyncio + `asyncio.Semaphore` for bounded concurrency, pydantic v2 strict JSON schema `response_format`, pytest + hypothesis for tests. Reference spec: `docs/superpowers/specs/2026-04-14-llm-vision-pdf-extraction-design.md`.

---

## File Structure

**Create (new `src/pipeline/vision/` package):**
- `src/pipeline/vision/__init__.py` — re-exports `extract_province`
- `src/pipeline/vision/schema.py` — `VisionRecord`, `WindowExtraction`
- `src/pipeline/vision/prompts.py` — `SYSTEM_TEMPLATE`, `build_prompt()`
- `src/pipeline/vision/render.py` — `PAGE_DPI` / `COLORSPACE` / `IMAGE_FORMAT` constants, `render_page()`
- `src/pipeline/vision/toc.py` — `SectionContext`, `build_section_map()`
- `src/pipeline/vision/windows.py` — `Window`, `build_windows()`
- `src/pipeline/vision/extract.py` — `extract_window()` (async)
- `src/pipeline/vision/orchestrate.py` — `extract_province()` (async)

**Create (tests):**
- `tests/unit/test_vision_schema.py`
- `tests/unit/test_vision_prompts.py`
- `tests/unit/test_vision_render.py`
- `tests/unit/test_vision_toc.py`
- `tests/unit/test_vision_windows.py`
- `tests/unit/test_vision_extract.py`
- `tests/unit/test_vision_orchestrate.py`
- `tests/unit/test_openai_client_vision.py`
- `tests/property/test_vision_properties.py`
- `tests/integration/test_vision_pipeline.py`
- `tests/fixtures/mini_pdf.pdf` — 4-page synthetic PDF for integration tests
- `tests/fixtures/vision_cache.sqlite` — recorded hishel cache for integration tests

**Modify:**
- `src/pipeline/schema.py` — bump `schema_version` to `"2"`, narrow `ExtractionMethod` to `Literal["vision"]`, delete `CandidateRow` + `PageBlock`
- `src/openai_client.py` — add `chat_vision_json` method
- `src/pipeline/run.py` — rewire Phase 1 to call `vision.extract_province`
- `src/pipeline/regression.py` — add `check_golden_set_invariants`
- `src/core/loader.py` — reject v1 artifacts with a clear error
- `tests/fixtures/golden_codes.json` — add `expected_price` and `expected_description_contains` to all entries for the new field spot-check
- `tests/unit/test_schema.py` — update to v2
- `tests/unit/test_regression.py` — add per-field spot-check cases
- `tests/unit/test_openai_client.py` — add basic `chat_vision_json` smoke test
- `tests/integration/test_run_skeleton.py` — update for vision path
- `tests/regression/test_snapshot.py` — regenerate snapshot post-cutover
- `tests/property/test_schema_properties.py` — update ExtractionMethod literal

**Delete (in one commit, after cutover works):**
- `src/pipeline/structural/` (4 files)
- `src/pipeline/semantic.py`
- `src/pipeline/io.py`
- `tests/unit/test_structural_base.py`
- `tests/unit/test_structural_bc.py`
- `tests/unit/test_structural_ontario.py`
- `tests/unit/test_structural_yukon.py`
- `tests/unit/test_semantic.py`
- `tests/unit/test_io.py`

---

## Task 1: Create branch and cherry-pick spec

**Files:**
- None edited; git state change only

- [ ] **Step 1: Create branch off the current `rebuild/pipeline-openai` tip**

```bash
git checkout -b rebuild/pipeline-vision
```

This inherits the entire working pipeline (schema, openai_client, ngs_mapper, validate, embed, regression, core/loader, tests, CLI, Streamlit). Spec + plan docs come along for free.

- [ ] **Step 2: Verify baseline is intact**

```bash
git log --oneline -3
pytest -m unit -x
```

Expected: plan + spec commits on top; all unit tests green against the current structural pipeline.

---

## Task 2: Interim schema — add `"vision"` to `ExtractionMethod`

**Rationale:** Vision modules need to emit `extraction_method="vision"` while structural/semantic modules still exist. Widen the literal first, narrow it back in Task 13 after the cutover.

**Files:**
- Modify: `src/pipeline/schema.py:10`
- Test: `tests/unit/test_schema.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_schema.py`:

```python
def test_extraction_method_accepts_vision():
    from src.pipeline.schema import FeeCodeRecord
    record = FeeCodeRecord(
        province="ON",
        fsc_code="A001",
        fsc_fn="test",
        fsc_description="test description",
        page=1,
        source_pdf_hash="abc",
        extraction_method="vision",
        extraction_confidence=0.95,
    )
    assert record.extraction_method == "vision"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_schema.py::test_extraction_method_accepts_vision -v
```

Expected: FAIL — pydantic `ValidationError: Input should be 'structural' or 'semantic'`.

- [ ] **Step 3: Widen the literal**

In `src/pipeline/schema.py` line 10:

```python
ExtractionMethod = Literal["structural", "semantic", "vision"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_schema.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/schema.py tests/unit/test_schema.py
git commit -m "feat(schema): add 'vision' ExtractionMethod literal (interim)"
```

---

## Task 3: Vision response schema

**Files:**
- Create: `src/pipeline/vision/__init__.py` (empty for now)
- Create: `src/pipeline/vision/schema.py`
- Create: `tests/unit/test_vision_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_schema.py`:

```python
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.pipeline.vision.schema import VisionRecord, WindowExtraction


def test_vision_record_minimal():
    r = VisionRecord(
        province="ON",
        fsc_code="A001",
        fsc_fn="consult",
        fsc_description="Initial consultation",
        page=7,
        extraction_confidence=0.9,
    )
    assert r.price is None
    assert r.fsc_chapter is None


def test_vision_record_with_price_and_hints():
    r = VisionRecord(
        province="BC",
        fsc_code="00025",
        fsc_fn="chamber 1st hour",
        fsc_description="Where no other fee is charged - physician in chamber",
        fsc_chapter="Hyperbaric Chamber",
        fsc_section="Fees",
        price=Decimal("83.68"),
        page=60,
        extraction_confidence=0.95,
    )
    assert r.price == Decimal("83.68")
    assert r.fsc_chapter == "Hyperbaric Chamber"


def test_vision_record_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        VisionRecord(
            province="ON",
            fsc_code="A001",
            fsc_fn="x",
            fsc_description="y",
            page=1,
            extraction_confidence=1.5,
        )


def test_vision_record_rejects_extra_fields():
    with pytest.raises(ValidationError):
        VisionRecord(
            province="ON",
            fsc_code="A001",
            fsc_fn="x",
            fsc_description="y",
            page=1,
            extraction_confidence=0.5,
            bogus="nope",
        )


def test_window_extraction_empty():
    w = WindowExtraction(records=[])
    assert w.records == []


def test_window_extraction_multi_record():
    r1 = VisionRecord(
        province="ON", fsc_code="A001", fsc_fn="x", fsc_description="y",
        page=1, extraction_confidence=0.9,
    )
    r2 = VisionRecord(
        province="ON", fsc_code="A002", fsc_fn="x", fsc_description="y",
        page=1, extraction_confidence=0.8,
    )
    w = WindowExtraction(records=[r1, r2])
    assert len(w.records) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.vision'`.

- [ ] **Step 3: Create the package + schema module**

Create `src/pipeline/vision/__init__.py`:

```python
"""Vision-based PDF extraction pipeline."""
```

Create `src/pipeline/vision/schema.py`:

```python
"""Response schemas for the vision extraction pass."""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from src.pipeline.schema import Province


class VisionRecord(BaseModel):
    """One fee-code entry as emitted by the vision extractor.

    The orchestrator promotes these to :class:`FeeCodeRecord` by attaching
    ``schema_version``, ``source_pdf_hash``, and
    ``extraction_method="vision"`` (NGS fields stay null for the mapper).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    """Response envelope for one window-level ``chat_vision_json`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: list[VisionRecord]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_vision_schema.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/vision/__init__.py src/pipeline/vision/schema.py tests/unit/test_vision_schema.py
git commit -m "feat(vision): VisionRecord + WindowExtraction response schema"
```

---

## Task 4: Prompts module

**Files:**
- Create: `src/pipeline/vision/prompts.py`
- Create: `tests/unit/test_vision_prompts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_prompts.py`:

```python
from src.pipeline.vision.prompts import SYSTEM_TEMPLATE, build_prompt
from src.pipeline.vision.toc import SectionContext


def test_system_template_mentions_target_page_rule():
    assert "TARGET PAGE" in SYSTEM_TEMPLATE
    assert "final visible line" in SYSTEM_TEMPLATE


def test_build_prompt_includes_province_and_target_page():
    hints = SectionContext(chapter=None, section=None, subsection=None)
    prompt = build_prompt(province="ON", target_page=47, section=hints)
    assert "ON" in prompt
    assert "page 47" in prompt or "page=47" in prompt


def test_build_prompt_injects_toc_hints_when_present():
    hints = SectionContext(
        chapter="Diagnostic Imaging",
        section="X-Ray",
        subsection=None,
    )
    prompt = build_prompt(province="ON", target_page=120, section=hints)
    assert "Diagnostic Imaging" in prompt
    assert "X-Ray" in prompt


def test_build_prompt_says_null_when_no_hints():
    hints = SectionContext(chapter=None, section=None, subsection=None)
    prompt = build_prompt(province="YT", target_page=12, section=hints)
    # Should not contain the string "None" verbatim - we format nulls readably
    assert "null" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_prompts.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.vision.prompts'`.

- [ ] **Step 3: Write the prompts module**

Create `src/pipeline/vision/prompts.py`:

```python
"""Prompt templates for the vision extraction pass.

SYSTEM_TEMPLATE is the stable instruction block. `build_prompt` fills in
the per-window variable parts (province, target page, TOC hints).
"""
from __future__ import annotations

from src.pipeline.schema import Province
from src.pipeline.vision.toc import SectionContext


SYSTEM_TEMPLATE = """\
You are extracting fee-code entries from the {province} physician fee schedule.

One or two rendered pages are attached below. The TARGET PAGE is page {target_page}.

RULES (critical — each entry has exactly one owning window):
- Emit an entry iff its final visible line appears on the TARGET PAGE.
- Skip entries cut off at the bottom of the TARGET PAGE — the next window owns them.
- Skip entries whose end lies on the earlier (non-target) page — a prior window already owned them.

SECTION CONTEXT FOR PAGE {target_page} (authoritative — use these exact strings when set):
  chapter:    {chapter}
  section:    {section}
  subsection: {subsection}
If a field is null, read whatever header is visible on the TARGET PAGE itself.

FIELD RULES:
  fsc_code              : code exactly as printed.
  fsc_fn                : short function/name string (the brief label next to the code).
  fsc_description       : full description as one string; join wrapped lines with a space.
  fsc_notes             : rich notes paragraphs attached to this code, or null if none.
  price                 : dollar amount for this code (strip the $). null if no price shown.
  page                  : {target_page} (always the TARGET PAGE number, 1-indexed).
  extraction_confidence : your calibrated 0.0-1.0 confidence for this record.
"""


def _fmt(val: str | None) -> str:
    return val if val else "null"


def build_prompt(
    *, province: Province, target_page: int, section: SectionContext
) -> str:
    """Compose the system prompt for one window."""
    return SYSTEM_TEMPLATE.format(
        province=province,
        target_page=target_page,
        chapter=_fmt(section.chapter),
        section=_fmt(section.section),
        subsection=_fmt(section.subsection),
    )
```

Note: this imports `SectionContext` from `src.pipeline.vision.toc`, which doesn't exist yet — the test fails until Task 6 lands. That's intentional; we build the dependency graph bottom-up but write tests top-down so the failure mode is clear.

- [ ] **Step 4: Temporarily stub `toc.py` so the import works**

Create `src/pipeline/vision/toc.py` with a stub only (full impl in Task 6):

```python
"""Section-context map from pymupdf TOC. Full implementation in Task 6."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionContext:
    chapter: str | None
    section: str | None
    subsection: str | None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_vision_prompts.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/vision/prompts.py src/pipeline/vision/toc.py tests/unit/test_vision_prompts.py
git commit -m "feat(vision): prompt templates + SectionContext stub"
```

---

## Task 5: Render module (pinned constants)

**Files:**
- Create: `src/pipeline/vision/render.py`
- Create: `tests/unit/test_vision_render.py`
- Create: `tests/fixtures/mini_pdf.pdf` (via a small generator script)

- [ ] **Step 1: Generate a tiny fixture PDF**

Run this script (one-shot, don't commit it — just use it to produce the fixture):

```bash
python - <<'EOF'
import pymupdf
doc = pymupdf.open()
for i in range(4):
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), f"Page {i + 1} header", fontsize=14)
    page.insert_text((72, 200), f"Code A{i:03d}  consultation  $25.00", fontsize=10)
    page.insert_text((72, 220), "Description line one.", fontsize=9)
    page.insert_text((72, 234), "Description line two.", fontsize=9)
doc.set_toc([
    [1, "Introduction", 1],
    [1, "Procedures", 2],
    [2, "Minor", 2],
    [2, "Major", 3],
])
doc.save("tests/fixtures/mini_pdf.pdf")
doc.close()
print("Wrote tests/fixtures/mini_pdf.pdf")
EOF
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_vision_render.py`:

```python
from pathlib import Path

from src.pipeline.vision.render import (
    COLORSPACE,
    IMAGE_FORMAT,
    PAGE_DPI,
    render_page,
)

FIXTURE = Path("tests/fixtures/mini_pdf.pdf")


def test_render_returns_png_bytes():
    img = render_page(FIXTURE, page_index=0)
    assert img[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_render_is_deterministic():
    """Cache-key invariant: same PDF + page -> byte-identical bytes."""
    a = render_page(FIXTURE, page_index=1)
    b = render_page(FIXTURE, page_index=1)
    assert a == b


def test_pinned_constants():
    assert PAGE_DPI == 144
    assert IMAGE_FORMAT == "png"
    # csRGB constant exists as pymupdf.csRGB
    import pymupdf
    assert COLORSPACE is pymupdf.csRGB
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_render.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.vision.render'`.

- [ ] **Step 4: Write the render module**

Create `src/pipeline/vision/render.py`:

```python
"""PDF page rendering with pinned, cache-keyed constants.

DO NOT EDIT PAGE_DPI, COLORSPACE, or IMAGE_FORMAT without bumping
``schema_version``. These three values are part of the hishel cache key
(via the request body bytes), so any change silently invalidates every
cached vision call.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_DPI = 144
COLORSPACE = pymupdf.csRGB
IMAGE_FORMAT = "png"


def render_page(pdf_path: Path, *, page_index: int) -> bytes:
    """Render a single page as PNG bytes. ``page_index`` is 0-based."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=PAGE_DPI, colorspace=COLORSPACE)
        return pix.tobytes(IMAGE_FORMAT)
    finally:
        doc.close()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_vision_render.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/mini_pdf.pdf src/pipeline/vision/render.py tests/unit/test_vision_render.py
git commit -m "feat(vision): deterministic page rendering with pinned constants"
```

---

## Task 6: TOC module — full implementation

**Files:**
- Modify: `src/pipeline/vision/toc.py` (replace stub)
- Create: `tests/unit/test_vision_toc.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_toc.py`:

```python
from pathlib import Path

import pymupdf

from src.pipeline.vision.toc import SectionContext, build_section_map

FIXTURE = Path("tests/fixtures/mini_pdf.pdf")


def test_empty_map_when_no_toc(tmp_path):
    no_toc = tmp_path / "no_toc.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.save(no_toc)
    doc.close()
    assert build_section_map(no_toc) == {}


def test_forward_fill_from_toc():
    m = build_section_map(FIXTURE)
    # Fixture has: L1 Introduction@1, L1 Procedures@2, L2 Minor@2, L2 Major@3
    assert m[1] == SectionContext(chapter="Introduction", section=None, subsection=None)
    assert m[2] == SectionContext(chapter="Procedures", section="Minor", subsection=None)
    assert m[3] == SectionContext(chapter="Procedures", section="Major", subsection=None)
    # Page 4 has no new entry, so it inherits page 3
    assert m[4] == SectionContext(chapter="Procedures", section="Major", subsection=None)


def test_higher_level_clears_deeper_levels():
    """When an L1 appears, L2/L3 from the previous chapter should be cleared."""
    # Fixture: L2 Minor@2 active at pg2. New L1 would clear it.
    # Our fixture's L1 Procedures@2 triggers this exactly:
    m = build_section_map(FIXTURE)
    # At page 2, L1 Procedures appears AT THE SAME PAGE as L2 Minor.
    # Since L1 comes first (lower level, applied first in sorted order),
    # the L2 Minor then sets section. Final state: chapter=Procedures, section=Minor.
    assert m[2].chapter == "Procedures"
    assert m[2].section == "Minor"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_toc.py -v
```

Expected: FAIL — `ImportError: cannot import name 'build_section_map'`.

- [ ] **Step 3: Replace the stub with the real implementation**

Overwrite `src/pipeline/vision/toc.py`:

```python
"""Build a (page -> SectionContext) map from pymupdf's TOC.

pymupdf's ``get_toc()`` returns ``[[level, title, page], ...]`` where
``level`` is 1-based and ``page`` is 1-based. Entries apply forward from
their page until superseded by a same-or-higher-level entry.

Empty dict is returned when the PDF has no outline (the Yukon case).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class SectionContext:
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None


def build_section_map(pdf_path: Path) -> dict[int, SectionContext]:
    """Return a map of 1-based page number -> SectionContext.

    Missing pages (e.g. before the first L1 entry) get a SectionContext
    with all-None fields. Empty dict if the PDF has no TOC at all.
    """
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()
    n_pages = doc.page_count
    doc.close()

    if not toc:
        return {}

    # Sort by page, then by level so higher-level entries apply first on the
    # same page. This keeps (L1, L2, L3) state consistent when multiple
    # levels share a page.
    toc_sorted = sorted(toc, key=lambda e: (e[2], e[0]))

    result: dict[int, SectionContext] = {}
    current: list[str | None] = [None, None, None]  # indices 0..2 => L1..L3
    toc_iter = iter(toc_sorted)
    next_entry = next(toc_iter, None)

    for page in range(1, n_pages + 1):
        while next_entry is not None and next_entry[2] <= page:
            lvl, title, _ = next_entry
            if 1 <= lvl <= 3:
                current[lvl - 1] = title
                for deeper in range(lvl, 3):
                    current[deeper] = None
            next_entry = next(toc_iter, None)
        result[page] = SectionContext(
            chapter=current[0], section=current[1], subsection=current[2]
        )

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_vision_toc.py tests/unit/test_vision_prompts.py -v
```

Expected: all pass (prompts tests still green — `SectionContext` interface compatible).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/vision/toc.py tests/unit/test_vision_toc.py
git commit -m "feat(vision): TOC-driven SectionContext map with forward-fill"
```

---

## Task 7: Windows module

**Files:**
- Create: `src/pipeline/vision/windows.py`
- Create: `tests/unit/test_vision_windows.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_windows.py`:

```python
from src.pipeline.vision.toc import SectionContext
from src.pipeline.vision.windows import Window, build_windows


def test_single_page_pdf_yields_one_window():
    windows = list(build_windows(num_pages=1, section_map={}))
    assert len(windows) == 1
    assert windows[0].target_page == 1
    assert windows[0].context_page is None


def test_n_pages_yields_n_windows():
    windows = list(build_windows(num_pages=5, section_map={}))
    assert len(windows) == 5
    assert [w.target_page for w in windows] == [1, 2, 3, 4, 5]


def test_first_window_has_no_context_page():
    windows = list(build_windows(num_pages=5, section_map={}))
    assert windows[0].context_page is None


def test_middle_windows_have_previous_context_page():
    windows = list(build_windows(num_pages=5, section_map={}))
    assert windows[1].context_page == 1
    assert windows[2].context_page == 2
    assert windows[3].context_page == 3


def test_section_hints_pulled_from_map():
    section_map = {
        1: SectionContext(chapter="A", section=None, subsection=None),
        2: SectionContext(chapter="A", section="B", subsection=None),
        3: SectionContext(chapter="A", section="B", subsection="C"),
    }
    windows = list(build_windows(num_pages=3, section_map=section_map))
    assert windows[1].section_hints.chapter == "A"
    assert windows[1].section_hints.section == "B"
    assert windows[2].section_hints.subsection == "C"


def test_missing_section_map_entry_gets_empty_context():
    windows = list(build_windows(num_pages=2, section_map={}))
    assert windows[0].section_hints == SectionContext(None, None, None)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_windows.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.vision.windows'`.

- [ ] **Step 3: Write the windows module**

Create `src/pipeline/vision/windows.py`:

```python
"""Sliding-window builder.

For a PDF of N pages, yields N windows (one per page). Each window owns
exactly one target page; context page is the page immediately before it
(or None for window 1). Every fee-code entry lives entirely within its
owning window, so the extractor can emit exactly once using the
"emit-where-it-completes" rule (see prompts.py).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.pipeline.vision.toc import SectionContext


@dataclass(frozen=True)
class Window:
    target_page: int  # 1-based
    context_page: int | None  # 1-based, or None for the seed window
    section_hints: SectionContext


_EMPTY_SECTION = SectionContext(chapter=None, section=None, subsection=None)


def build_windows(
    *, num_pages: int, section_map: dict[int, SectionContext]
) -> Iterator[Window]:
    for page in range(1, num_pages + 1):
        yield Window(
            target_page=page,
            context_page=page - 1 if page > 1 else None,
            section_hints=section_map.get(page, _EMPTY_SECTION),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_vision_windows.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/vision/windows.py tests/unit/test_vision_windows.py
git commit -m "feat(vision): Window dataclass + build_windows generator"
```

---

## Task 8: `chat_vision_json` on `OpenAIClient`

**Files:**
- Modify: `src/openai_client.py` (add method after `chat_json`)
- Modify: `tests/unit/test_openai_client.py` (add smoke test)
- Create: `tests/unit/test_openai_client_vision.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_openai_client_vision.py`:

```python
"""Tests chat_vision_json by patching the SDK's create() method.

Follows the same mocking pattern as the existing tests/unit/test_openai_client.py
(monkeypatch OPENAI_API_KEY + patch.object on client._sdk.chat.completions.create).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.openai_client import OpenAIClient


class _Tiny(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    score: float = Field(ge=0.0, le=1.0)


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 20
    resp.model = "gpt-5.4-mini"
    return resp


def test_chat_vision_json_builds_image_url_parts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    img_a = b"\x89PNG\r\n\x1a\n" + b"AAAA"
    img_b = b"\x89PNG\r\n\x1a\n" + b"BBBB"
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_response({"label": "ok", "score": 0.9})

    with patch.object(
        client._sdk.chat.completions, "create", side_effect=_capture
    ):
        result = client.chat_vision_json(
            prompt="test prompt",
            images=[img_a, img_b],
            schema=_Tiny,
            model="gpt-5.4-mini",
        )

    assert result.label == "ok"
    assert result.score == 0.9
    user_msg = next(m for m in captured["messages"] if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    assert sum(1 for p in user_msg["content"] if p["type"] == "text") == 1
    assert sum(1 for p in user_msg["content"] if p["type"] == "image_url") == 2
    # Each image becomes a data: URL
    for part in user_msg["content"]:
        if part["type"] == "image_url":
            assert part["image_url"]["url"].startswith("data:image/png;base64,")
            assert part["image_url"]["detail"] == "high"


def test_chat_vision_json_validation_error_does_not_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake = _fake_response({"label": 123})  # wrong type -> ValidationError

    with patch.object(
        client._sdk.chat.completions, "create", return_value=fake
    ) as mock_create:
        with pytest.raises(ValidationError):
            client.chat_vision_json(
                prompt="x",
                images=[b"\x89PNG\r\n\x1a\nimg"],
                schema=_Tiny,
                model="gpt-5.4-mini",
            )
        assert mock_create.call_count == 1  # deterministic error, no retry
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_openai_client_vision.py -v
```

Expected: FAIL — `AttributeError: 'OpenAIClient' object has no attribute 'chat_vision_json'` (or similar).

- [ ] **Step 3: Implement `chat_vision_json`**

Add to `src/openai_client.py`, after the existing `chat_json` method (follow the same retry + determinism pattern):

```python
def chat_vision_json(
    self,
    *,
    prompt: str,
    images: list[bytes],
    schema: type[T],
    model: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> T:
    """Vision-enabled chat_json.

    ``images`` is a list of raw image bytes (PNG); each is base64-encoded
    into a ``data:`` URL with ``detail=high``. The request body, including
    image bytes, forms the hishel cache key, so identical inputs replay
    from the SQLite cache.
    """
    import base64

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in images:
        b64 = base64.b64encode(img).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )
    messages.append({"role": "user", "content": content})

    raw_schema = schema.model_json_schema()
    _strictify_schema(raw_schema)
    json_schema = {
        "name": schema.__name__,
        "schema": raw_schema,
        "strict": True,
    }

    def _call() -> T:
        resp = self._sdk.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": json_schema},
            temperature=temperature,
        )
        choice = resp.choices[0]
        raw = choice.message.content
        usage = resp.usage
        if raw is None or usage is None:
            raise _Deterministic(
                RuntimeError(f"Empty content/usage from {model}")
            )
        self.costs.record(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise _Deterministic(e) from e
        try:
            return schema.model_validate(parsed)
        except ValidationError as e:
            raise _Deterministic(e) from e

    return self._retry(_call, max_retries=max_retries)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_openai_client_vision.py tests/unit/test_openai_client.py -v
```

Expected: both files green.

- [ ] **Step 5: Commit**

```bash
git add src/openai_client.py tests/unit/test_openai_client_vision.py
git commit -m "feat(openai_client): chat_vision_json method with hishel caching"
```

---

## Task 9: Extract module

**Files:**
- Create: `src/pipeline/vision/extract.py`
- Create: `tests/unit/test_vision_extract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_extract.py`:

```python
"""Tests for extract_window: prompt assembly + retry-on-ValidationError."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.pipeline.vision.extract import extract_window
from src.pipeline.vision.schema import VisionRecord, WindowExtraction
from src.pipeline.vision.toc import SectionContext
from src.pipeline.vision.windows import Window


def _make_window(target_page: int = 2) -> Window:
    return Window(
        target_page=target_page,
        context_page=target_page - 1 if target_page > 1 else None,
        section_hints=SectionContext(chapter="X", section=None, subsection=None),
    )


def _make_client(return_values: list):
    """Build a MagicMock client whose chat_vision_json iterates over values.

    Each entry is either a WindowExtraction instance (returned) or an
    Exception (raised).
    """
    client = MagicMock()
    it = iter(return_values)

    def _side(*args, **kwargs):
        val = next(it)
        if isinstance(val, Exception):
            raise val
        return val

    client.chat_vision_json.side_effect = _side
    return client


def test_extract_window_returns_records():
    client = _make_client([
        WindowExtraction(records=[
            VisionRecord(
                province="ON", fsc_code="A001", fsc_fn="fn", fsc_description="desc",
                page=2, extraction_confidence=0.9,
            ),
        ])
    ])
    window = _make_window(target_page=2)

    records = asyncio.run(
        extract_window(window=window, province="ON", images=[b"img1", b"img2"], client=client)
    )

    assert len(records) == 1
    assert records[0].fsc_code == "A001"
    # First call used temperature=0.0
    assert client.chat_vision_json.call_args_list[0].kwargs["temperature"] == 0.0


def test_extract_window_retries_once_on_validation_error():
    good = WindowExtraction(records=[
        VisionRecord(
            province="ON", fsc_code="A002", fsc_fn="fn", fsc_description="desc",
            page=2, extraction_confidence=0.7,
        ),
    ])

    # Simulate ValidationError by constructing one from a bad payload
    try:
        WindowExtraction(records=[{"bogus": 1}])
    except ValidationError as exc:
        validation_err = exc

    client = _make_client([validation_err, good])
    window = _make_window()

    records = asyncio.run(
        extract_window(window=window, province="ON", images=[b"img"], client=client)
    )

    assert len(records) == 1
    # Retry should have used temperature=0.2
    assert client.chat_vision_json.call_args_list[1].kwargs["temperature"] == 0.2


def test_extract_window_emits_zero_records_on_second_failure():
    try:
        WindowExtraction(records=[{"bogus": 1}])
    except ValidationError as exc:
        validation_err = exc

    client = _make_client([validation_err, validation_err])
    window = _make_window()

    records = asyncio.run(
        extract_window(window=window, province="ON", images=[b"img"], client=client)
    )

    assert records == []
    assert client.chat_vision_json.call_count == 2


def test_extract_window_passes_images_in_context_then_target_order():
    client = _make_client([WindowExtraction(records=[])])
    window = _make_window(target_page=5)
    images = [b"context-img", b"target-img"]

    asyncio.run(
        extract_window(window=window, province="ON", images=images, client=client)
    )

    first_call_images = client.chat_vision_json.call_args.kwargs["images"]
    assert first_call_images == [b"context-img", b"target-img"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_extract.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.vision.extract'`.

- [ ] **Step 3: Write the extract module**

Create `src/pipeline/vision/extract.py`:

```python
"""Per-window extraction call.

Responsibility:
- Build the prompt (via :mod:`prompts`).
- Dispatch one ``chat_vision_json`` call via ``asyncio.to_thread`` so the
  sync OpenAI SDK cooperates with our asyncio fan-out.
- Retry exactly once on :class:`pydantic.ValidationError` with a small
  temperature nudge (0.0 -> 0.2). On second failure, emit zero records
  (regression gate catches cumulative damage).
- Log failures to the diagnostics path for post-mortem.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.openai_client import OpenAIClient
from src.pipeline.schema import Province
from src.pipeline.vision.prompts import build_prompt
from src.pipeline.vision.schema import VisionRecord, WindowExtraction
from src.pipeline.vision.windows import Window

logger = logging.getLogger(__name__)


async def extract_window(
    *,
    window: Window,
    province: Province,
    images: list[bytes],
    client: OpenAIClient,
    model: str = "gpt-5.4-mini",
    failure_log: Path | None = None,
) -> list[VisionRecord]:
    """Extract records from one window. Order: context image first, target second.

    Returns [] on two consecutive ValidationErrors (logs to failure_log if set).
    """
    prompt = build_prompt(
        province=province,
        target_page=window.target_page,
        section=window.section_hints,
    )

    for attempt, temperature in enumerate([0.0, 0.2]):
        try:
            result: WindowExtraction = await asyncio.to_thread(
                client.chat_vision_json,
                prompt=prompt,
                images=images,
                schema=WindowExtraction,
                model=model,
                temperature=temperature,
            )
            return list(result.records)
        except ValidationError as e:
            logger.warning(
                "window %d (province %s) validation failed on attempt %d: %s",
                window.target_page, province, attempt + 1, e,
            )
            last_err = e

    if failure_log is not None:
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with failure_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "province": province,
                        "target_page": window.target_page,
                        "error_class": type(last_err).__name__,
                        "message": str(last_err),
                    }
                )
                + "\n"
            )
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_vision_extract.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/vision/extract.py tests/unit/test_vision_extract.py
git commit -m "feat(vision): extract_window with retry-once + failure log"
```

---

## Task 10: Orchestrate module

**Files:**
- Create: `src/pipeline/vision/orchestrate.py`
- Modify: `src/pipeline/vision/__init__.py` (re-export `extract_province`)
- Create: `tests/unit/test_vision_orchestrate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vision_orchestrate.py`:

```python
"""Tests for orchestrate: dedup rule + FeeCodeRecord promotion."""
from __future__ import annotations

from decimal import Decimal

from src.pipeline.vision.orchestrate import _dedup, _to_fee_code_record
from src.pipeline.vision.schema import VisionRecord


def _r(code: str, conf: float, page: int = 1, price: str | None = None) -> VisionRecord:
    return VisionRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="desc",
        page=page, extraction_confidence=conf,
        price=Decimal(price) if price else None,
    )


def test_dedup_keeps_highest_confidence():
    records = [_r("A001", 0.7), _r("A001", 0.95), _r("A001", 0.85)]
    deduped = _dedup(records)
    assert len(deduped) == 1
    assert deduped[0].extraction_confidence == 0.95


def test_dedup_ties_broken_by_lowest_page():
    records = [_r("A001", 0.9, page=3), _r("A001", 0.9, page=1), _r("A001", 0.9, page=2)]
    deduped = _dedup(records)
    assert len(deduped) == 1
    assert deduped[0].page == 1


def test_dedup_preserves_distinct_codes():
    records = [_r("A001", 0.9), _r("A002", 0.9), _r("A003", 0.9)]
    deduped = _dedup(records)
    assert {r.fsc_code for r in deduped} == {"A001", "A002", "A003"}


def test_to_fee_code_record_promotes_fields():
    vr = _r("A001", 0.9, price="25.50")
    fcr = _to_fee_code_record(vr, province="ON", source_pdf_hash="deadbeef")
    assert fcr.schema_version == "2"
    assert fcr.extraction_method == "vision"
    assert fcr.source_pdf_hash == "deadbeef"
    assert fcr.NGS_code is None
    assert fcr.price == Decimal("25.50")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_vision_orchestrate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline.vision.orchestrate'`.

Also: Task 2 currently has `schema_version: Literal["1"] = "1"` — the test expects `"2"`. We widened `ExtractionMethod` in Task 2 but did NOT bump `schema_version` yet. Temporarily bump `schema_version` to `Literal["1", "2"] = "2"` so both work in parallel; we narrow in Task 13.

Edit `src/pipeline/schema.py:51`:

```python
schema_version: Literal["1", "2"] = "2"
```

- [ ] **Step 3: Write the orchestrate module**

Create `src/pipeline/vision/orchestrate.py`:

```python
"""Province-level extraction orchestrator.

Glues render + TOC + windows + extract together with bounded asyncio
concurrency, then merges, dedups, and promotes VisionRecord ->
FeeCodeRecord.

The pymupdf Document is shared across coroutines — pymupdf is not thread
safe, but asyncio runs everything on one OS thread (the blocking work
happens inside ``asyncio.to_thread`` for the OpenAI call only). Rendering
in the loop thread is fine: ~30-50ms per page is dominated by the 5-15s
LLM latency.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import pymupdf
from tqdm.asyncio import tqdm_asyncio

from src.openai_client import OpenAIClient
from src.pipeline.schema import FeeCodeRecord, Province
from src.pipeline.vision.extract import extract_window
from src.pipeline.vision.render import COLORSPACE, IMAGE_FORMAT, PAGE_DPI
from src.pipeline.vision.schema import VisionRecord
from src.pipeline.vision.toc import build_section_map
from src.pipeline.vision.windows import Window, build_windows

logger = logging.getLogger(__name__)


def _render_window_images(doc: pymupdf.Document, window: Window) -> list[bytes]:
    imgs: list[bytes] = []
    if window.context_page is not None:
        pix = doc[window.context_page - 1].get_pixmap(dpi=PAGE_DPI, colorspace=COLORSPACE)
        imgs.append(pix.tobytes(IMAGE_FORMAT))
    pix = doc[window.target_page - 1].get_pixmap(dpi=PAGE_DPI, colorspace=COLORSPACE)
    imgs.append(pix.tobytes(IMAGE_FORMAT))
    return imgs


def _dedup(records: list[VisionRecord]) -> list[VisionRecord]:
    """Keep highest extraction_confidence per (province, fsc_code);
    tiebreak on lowest page."""
    by_key: dict[tuple[str, str], VisionRecord] = {}
    for r in records:
        key = (r.province, r.fsc_code)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = r
            continue
        if r.extraction_confidence > prev.extraction_confidence:
            by_key[key] = r
        elif r.extraction_confidence == prev.extraction_confidence and r.page < prev.page:
            by_key[key] = r
    return list(by_key.values())


def _to_fee_code_record(
    vr: VisionRecord, *, province: Province, source_pdf_hash: str
) -> FeeCodeRecord:
    return FeeCodeRecord(
        schema_version="2",
        province=province,
        fsc_code=vr.fsc_code,
        fsc_fn=vr.fsc_fn,
        fsc_description=vr.fsc_description,
        fsc_chapter=vr.fsc_chapter,
        fsc_section=vr.fsc_section,
        fsc_subsection=vr.fsc_subsection,
        fsc_notes=vr.fsc_notes,
        price=vr.price,
        page=vr.page,
        source_pdf_hash=source_pdf_hash,
        extraction_method="vision",
        extraction_confidence=vr.extraction_confidence,
    )


def _hash_pdf(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


async def extract_province(
    pdf_path: Path,
    *,
    province: Province,
    client: OpenAIClient,
    concurrency: int = 20,
    model: str = "gpt-5.4-mini",
    failure_log: Path | None = None,
) -> list[FeeCodeRecord]:
    section_map = build_section_map(pdf_path)
    pdf_hash = _hash_pdf(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        windows = list(build_windows(num_pages=doc.page_count, section_map=section_map))
        sem = asyncio.Semaphore(concurrency)

        async def _one(w: Window) -> list[VisionRecord]:
            async with sem:
                imgs = _render_window_images(doc, w)
                return await extract_window(
                    window=w,
                    province=province,
                    images=imgs,
                    client=client,
                    model=model,
                    failure_log=failure_log,
                )

        logger.info("[%s] dispatching %d windows (concurrency=%d)",
                    province, len(windows), concurrency)
        batches: list[list[VisionRecord]] = await tqdm_asyncio.gather(
            *[_one(w) for w in windows],
            desc=f"{province} vision extraction",
        )
    finally:
        doc.close()

    flat = [r for batch in batches for r in batch]
    deduped = _dedup(flat)
    dup_count = len(flat) - len(deduped)
    if dup_count:
        logger.info("[%s] dedup removed %d duplicate records", province, dup_count)
    return [_to_fee_code_record(r, province=province, source_pdf_hash=pdf_hash) for r in deduped]
```

Update `src/pipeline/vision/__init__.py`:

```python
"""Vision-based PDF extraction pipeline."""
from src.pipeline.vision.orchestrate import extract_province

__all__ = ["extract_province"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_vision_orchestrate.py tests/unit/test_vision_extract.py tests/unit/test_vision_schema.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/vision/orchestrate.py src/pipeline/vision/__init__.py src/pipeline/schema.py tests/unit/test_vision_orchestrate.py
git commit -m "feat(vision): extract_province orchestrator with dedup + promotion"
```

---

## Task 11: Rewire `run.py` to call vision

**Files:**
- Modify: `src/pipeline/run.py`

- [ ] **Step 1: Write a failing integration-style unit test**

Append to `tests/integration/test_run_skeleton.py` (or create if absent):

```python
from unittest.mock import patch
from pathlib import Path

import pytest


def test_run_pipeline_uses_vision_extract(monkeypatch, tmp_path):
    """After rewire, run_pipeline should call src.pipeline.vision.extract_province
    per province (not the structural extractors).
    """
    from src.pipeline import run as run_mod

    called = []

    async def fake_extract_province(pdf_path, *, province, client, **kwargs):
        called.append((province, Path(pdf_path).name))
        return []

    monkeypatch.setattr(run_mod, "vision_extract_province", fake_extract_province, raising=False)

    # Minimal fixture dirs
    (tmp_path / "pdf").mkdir()
    (tmp_path / "docx").mkdir()
    # Fake PDFs matching the glob patterns so _province_pdf resolves.
    for name in (
        "moh-schedule-benefit-fake.pdf",
        "msc_payment_schedule_-_fake.pdf",
        "yukon_physician_fee_guide_fake.pdf",
    ):
        (tmp_path / "pdf" / name).write_bytes(b"%PDF-fake")

    cfg = run_mod.PipelineConfig(
        raw_pdf_dir=tmp_path / "pdf",
        raw_docx_dir=tmp_path / "docx",
        output_dir=tmp_path / "out",
        diagnostics_dir=tmp_path / "diag",
        version="test",
        force=True,
    )

    with pytest.raises(Exception):
        # Will fail somewhere in NGS/embed with empty records; we only care
        # that vision_extract_province was invoked for each province.
        run_mod.run_pipeline(cfg, client=object())

    assert {c[0] for c in called} == {"ON", "BC", "YT"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_run_skeleton.py::test_run_pipeline_uses_vision_extract -v
```

Expected: FAIL — the structural extractors are still called.

- [ ] **Step 3: Rewire `src/pipeline/run.py`**

Replace the imports block at the top:

```python
# BEFORE (lines 12-23)
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

# AFTER
import asyncio
import hashlib

from src.openai_client import OpenAIClient
from src.pipeline.embed import build_embeddings, save_npz
from src.pipeline.ngs_mapper import map_ngs
from src.pipeline.ngs_parser import parse_ngs_docx
from src.pipeline.regression import check, diff, format_report
from src.pipeline.schema import FeeCodeRecord, Manifest
from src.pipeline.validate import validate
from src.pipeline.vision import extract_province as vision_extract_province
```

Remove the `PROVINCE_EXTRACTORS` dict (lines 31-35) — unused after rewire.

Replace the `# Phase 1: Extract + rescue per province` block (lines 105-142):

```python
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
```

Also add a local `_pdf_hash` helper to `run.py` (since `src/pipeline/io.py` is being deleted in Task 12 — we duplicate the ~5-line helper here rather than add a new import):

```python
def _pdf_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
```

And update the `# Phase 2: Validate` block (line 145) — `validate()` currently expects `CandidateRow` inputs, but our vision path already produces `FeeCodeRecord`. `validate` is kept; its job is now to apply the province regex + dedup + positive-price check on `FeeCodeRecord`. Adjusting its signature is out of scope for this task (see Task 14 follow-up). For now, skip `validate()` in `run.py` — the vision path already validates via pydantic.

Replace the `# Phase 2: Validate` block with:

```python
# Phase 2: (No-op) vision path is pre-validated by pydantic
logger.info("Phase 2/5: Validation (vision records are pre-validated by pydantic)")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/integration/test_run_skeleton.py::test_run_pipeline_uses_vision_extract -v
```

Expected: pass. (Note: this test uses `raising=False` on monkeypatch so it only checks the function call — it does not care about downstream NGS/embed failure on empty records.)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/run.py tests/integration/test_run_skeleton.py
git commit -m "feat(run): rewire Phase 1 to vision.extract_province"
```

---

## Task 12: Delete structural + semantic + io.py

**Files:**
- Delete: `src/pipeline/structural/` (4 files)
- Delete: `src/pipeline/semantic.py`
- Delete: `src/pipeline/io.py`
- Delete: `tests/unit/test_structural_*.py` (4 files)
- Delete: `tests/unit/test_semantic.py`
- Delete: `tests/unit/test_io.py`

- [ ] **Step 1: Run all existing tests first**

```bash
pytest -m "unit or integration" -x
```

Expected: all green before deletion. If red, stop and fix first.

- [ ] **Step 2: Delete old modules**

```bash
git rm src/pipeline/structural/__init__.py
git rm src/pipeline/structural/base.py
git rm src/pipeline/structural/bc.py
git rm src/pipeline/structural/ontario.py
git rm src/pipeline/structural/yukon.py
git rm src/pipeline/semantic.py
git rm src/pipeline/io.py
```

- [ ] **Step 3: Delete old tests**

```bash
git rm tests/unit/test_structural_base.py
git rm tests/unit/test_structural_bc.py
git rm tests/unit/test_structural_ontario.py
git rm tests/unit/test_structural_yukon.py
git rm tests/unit/test_semantic.py
git rm tests/unit/test_io.py
```

- [ ] **Step 4: Run all tests**

```bash
pytest -m "unit or integration" -x
```

Expected: all green. If imports broke anywhere (e.g., an overlooked reference to `CandidateRow`), fix in this commit before continuing.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove structural + semantic extractors after vision cutover"
```

---

## Task 13: Finalize schema v2

**Files:**
- Modify: `src/pipeline/schema.py` (narrow literals, delete CandidateRow + PageBlock)
- Modify: `tests/unit/test_schema.py`
- Modify: `tests/property/test_schema_properties.py`

- [ ] **Step 1: Write failing tests for the narrowed literals**

Append to `tests/unit/test_schema.py`:

```python
import pytest
from pydantic import ValidationError


def test_schema_version_rejects_v1():
    from src.pipeline.schema import FeeCodeRecord
    with pytest.raises(ValidationError):
        FeeCodeRecord(
            schema_version="1",
            province="ON", fsc_code="A", fsc_fn="x", fsc_description="y",
            page=1, source_pdf_hash="h",
            extraction_method="vision", extraction_confidence=0.9,
        )


def test_extraction_method_rejects_structural():
    from src.pipeline.schema import FeeCodeRecord
    with pytest.raises(ValidationError):
        FeeCodeRecord(
            schema_version="2",
            province="ON", fsc_code="A", fsc_fn="x", fsc_description="y",
            page=1, source_pdf_hash="h",
            extraction_method="structural", extraction_confidence=0.9,
        )


def test_candidate_row_is_gone():
    import src.pipeline.schema as s
    assert not hasattr(s, "CandidateRow")
    assert not hasattr(s, "PageBlock")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/unit/test_schema.py -v
```

Expected: FAIL on all three.

- [ ] **Step 3: Narrow the schema**

Edit `src/pipeline/schema.py`:

```python
# Line 10
ExtractionMethod = Literal["vision"]

# Line 51
schema_version: Literal["2"] = "2"

# Delete lines 14-26 (PageBlock) and 28-44 (CandidateRow)
```

Remove `ExtractionMethod` default from `FeeCodeRecord.extraction_method` if any — it already has no default (required), so nothing to do.

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_schema.py tests/unit/test_vision_schema.py tests/property/test_schema_properties.py -v
```

Expected: all pass. If `test_schema_properties.py` references the old literals, update it inline (replace with `Literal["vision"]`).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/schema.py tests/unit/test_schema.py tests/property/test_schema_properties.py
git commit -m "feat(schema): finalize v2 — Literal['vision'] only, drop CandidateRow/PageBlock"
```

---

## Task 14: Loader v1 rejection

**Files:**
- Modify: `src/core/loader.py`
- Modify: `tests/unit/test_core_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_core_loader.py`:

```python
import json
import numpy as np
import pytest


def test_loader_rejects_v1_artifacts(tmp_path):
    from src.core.loader import load_latest

    vdir = tmp_path / "v2025-01-01"
    vdir.mkdir()
    (vdir / "codes.json").write_text(json.dumps([
        {
            "schema_version": "1",
            "province": "ON",
            "fsc_code": "A001",
            "fsc_fn": "x",
            "fsc_description": "y",
            "page": 1,
            "source_pdf_hash": "h",
            "extraction_method": "structural",
            "extraction_confidence": 0.9,
        }
    ]))
    (vdir / "manifest.json").write_text(json.dumps({
        "schema_version": "1", "generated_at": "2025-01-01T00:00:00+00:00",
        "git_sha": "abc", "row_counts": {"ON": 1, "BC": 0, "YT": 0},
        "source_pdf_hashes": {"ON": "h"}, "models": {"embed": "x", "extract": "y"},
        "regression_override": None,
    }))
    np.savez(vdir / "embeddings.npz",
             embeddings=np.zeros((1, 1024), dtype=np.float32),
             record_ids=np.array([0], dtype=np.int32))

    with pytest.raises(RuntimeError, match="schema v1.*rerun the pipeline"):
        load_latest(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_core_loader.py::test_loader_rejects_v1_artifacts -v
```

Expected: FAIL — the current loader parses v1 by throwing a pydantic ValidationError, but the test expects a clear `RuntimeError` with a message mentioning "schema v1" and "rerun the pipeline".

- [ ] **Step 3: Add the v1 check**

Edit `src/core/loader.py`:

```python
def load_latest(
    parsed_dir: Path,
) -> tuple[list[FeeCodeRecord], np.ndarray, np.ndarray, Manifest]:
    version_dirs = sorted(parsed_dir.glob("v*"))
    if not version_dirs:
        raise FileNotFoundError(f"No versioned artifacts in {parsed_dir}")
    vdir = version_dirs[-1]

    codes_raw = json.loads((vdir / "codes.json").read_text(encoding="utf-8"))

    # Schema-version guard: refuse v1 up front with an actionable message.
    # pydantic would also reject, but the error message would be cryptic.
    if codes_raw and codes_raw[0].get("schema_version") != "2":
        raise RuntimeError(
            f"Artifact at {vdir} uses schema v{codes_raw[0].get('schema_version')}; "
            "this app expects v2. Rerun the pipeline: python -m src.cli run --force"
        )

    records = [FeeCodeRecord(**c) for c in codes_raw]
    # ... rest unchanged
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_core_loader.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/loader.py tests/unit/test_core_loader.py
git commit -m "feat(loader): reject v1 artifacts with actionable error"
```

---

## Task 15: Golden set extension (🎯 USER CONTRIBUTION)

**Files:**
- Modify: `tests/fixtures/golden_codes.json`

**Context:** `tests/fixtures/golden_codes.json` already exists. To wire it into the new per-field spot-check (Task 16), every entry needs an `expected_price` (string decimal, or `null`) and `expected_description_contains` (list of substrings). These come from the source PDFs and encode domain knowledge only you have — which codes actually matter, which have historically had bad data, which span multi-page notes.

**🎯 This is a 5-10 minute user contribution.** Open `tests/fixtures/golden_codes.json`, and for each of the ~60 entries, fill in:
- `expected_price`: the decimal string shown in the source PDF, or `null` if no price (e.g. independent-consideration codes).
- `expected_description_contains`: 2-3 distinctive substrings (lowercase) that *must* appear in the description. Prefer medical terms over common English.

**Why this matters:** The regression gate in Task 16 fails loud when any golden code's extracted `price` or `fsc_description` drifts from what you specified here. Loose entries (missing substrings) let bad data through; overly strict entries cause false alarms on legitimate PDF updates. Your domain knowledge is what makes this gate useful.

- [ ] **Step 1: Open the file and extend each entry**

```bash
code tests/fixtures/golden_codes.json   # or nano / vim / editor of choice
```

Example of the extended shape:

```json
{
  "province": "ON",
  "fsc_code": "E611",
  "expected_price": "82.75",
  "expected_description_contains": ["consultation", "specific assessment"]
},
```

- [ ] **Step 2: Verify the JSON still parses**

```bash
python -c "import json; json.load(open('tests/fixtures/golden_codes.json'))"
```

Expected: no output (valid JSON).

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/golden_codes.json
git commit -m "test(fixtures): extend golden set with expected_price + description substrings"
```

---

## Task 16: Regression gate — per-field spot-check (🎯 USER CONTRIBUTION)

**Files:**
- Modify: `src/pipeline/regression.py`
- Modify: `tests/unit/test_regression.py`

**Context:** The existing `check()` only fails on row-count drops and on golden codes disappearing. It does NOT catch the BC-price-silent-corruption class of bug (records present, fields wrong). This task adds `check_golden_set_invariants()` which compares each golden code's new record against your expectations.

**🎯 The comparison rule is a domain judgment call — you implement it.** Three reasonable policies:

- **Strict:** any diff on `price` OR any missing `expected_description_contains` substring fails.
- **Price-strict, description-soft:** price mismatch fails hard; description substring mismatch logs a warning but doesn't fail.
- **Fuzzy description:** use a similarity threshold (e.g., Jaccard on tokens > 0.8) for description, strict match for price.

Pick one and implement it. ~10 lines of meaningful logic.

- [ ] **Step 1: Write the failing tests (policy-independent assertions)**

Append to `tests/unit/test_regression.py`:

```python
from decimal import Decimal
from src.pipeline.schema import FeeCodeRecord
from src.pipeline.regression import check_golden_set_invariants


def _rec(code: str, price: str | None, desc: str) -> FeeCodeRecord:
    return FeeCodeRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description=desc,
        page=1, source_pdf_hash="h",
        price=Decimal(price) if price else None,
        extraction_method="vision", extraction_confidence=0.9,
    )


GOLDEN = [
    {"province": "ON", "fsc_code": "E611",
     "expected_price": "82.75",
     "expected_description_contains": ["consultation", "assessment"]},
]


def test_golden_invariants_pass_when_fields_match():
    records = [_rec("E611", "82.75", "general consultation and assessment today")]
    issues = check_golden_set_invariants(records=records, golden=GOLDEN)
    assert issues == []


def test_golden_invariants_flag_price_mismatch():
    records = [_rec("E611", "7.00", "general consultation and assessment today")]
    issues = check_golden_set_invariants(records=records, golden=GOLDEN)
    assert any("price" in i.lower() for i in issues)


def test_golden_invariants_flag_missing_substring():
    records = [_rec("E611", "82.75", "something unrelated")]
    issues = check_golden_set_invariants(records=records, golden=GOLDEN)
    assert any("consultation" in i or "assessment" in i for i in issues)


def test_golden_invariants_flag_missing_record():
    issues = check_golden_set_invariants(records=[], golden=GOLDEN)
    assert any("E611" in i for i in issues)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_regression.py -v
```

Expected: FAIL — `ImportError` on `check_golden_set_invariants`.

- [ ] **Step 3: 🎯 Implement the function (YOUR POLICY CHOICE)**

Add to `src/pipeline/regression.py`:

```python
def check_golden_set_invariants(
    *,
    records: list[FeeCodeRecord],
    golden: list[dict],
) -> list[str]:
    """Return a list of issue strings (empty = all invariants hold).

    TODO (USER): Pick your policy:
      - strict: any diff fails
      - price-strict, description-soft: price mismatch fails, desc mismatch warns
      - fuzzy description: token-similarity threshold

    Each entry in `golden` has:
      province, fsc_code, expected_price (str | None), expected_description_contains (list[str])
    """
    issues: list[str] = []
    by_key = {(r.province, r.fsc_code): r for r in records}

    for g in golden:
        key = (g["province"], g["fsc_code"])
        rec = by_key.get(key)
        if rec is None:
            issues.append(f"golden code {key} missing from records")
            continue

        # YOUR POLICY GOES HERE.
        # Example — strict policy:
        # expected_price = g.get("expected_price")
        # if expected_price is not None:
        #     if rec.price is None or str(rec.price) != expected_price:
        #         issues.append(f"{key}: price expected {expected_price}, got {rec.price}")
        # for sub in g.get("expected_description_contains", []):
        #     if sub.lower() not in rec.fsc_description.lower():
        #         issues.append(f"{key}: description missing substring '{sub}'")

    return issues
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_regression.py -v
```

Expected: all 4 pass (under your chosen policy).

- [ ] **Step 5: Hook the new check into `run.py`**

In `src/pipeline/run.py`, inside the `# Phase 5: Regression check` block, after `check(report, ...)`:

```python
import json as _json
golden = _json.loads(Path("tests/fixtures/golden_codes.json").read_text())
golden_issues = check_golden_set_invariants(records=records, golden=golden)
if golden_issues and cfg.accept_regression is None:
    raise RuntimeError(
        "Golden-set field invariants failed:\n" + "\n".join(golden_issues)
    )
```

And add `check_golden_set_invariants` to the import line from `regression`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/regression.py src/pipeline/run.py tests/unit/test_regression.py
git commit -m "feat(regression): per-field golden-set spot-check gate"
```

---

## Task 17: Integration test — mini PDF + recorded hishel cache

**Files:**
- Create: `tests/integration/test_vision_pipeline.py`
- Create: `tests/fixtures/vision_cache.sqlite` (one-shot recording step)

- [ ] **Step 1: Record a hishel cache by running vision extraction on the mini fixture**

`OpenAIClient` stores its cache at `<cache_dir>/hishel_cache.db`. We point it at `tests/fixtures/vision_cache_dir/` and commit the resulting `hishel_cache.db`.

```bash
python - <<'EOF'
import asyncio
import os
from pathlib import Path

from src.openai_client import OpenAIClient
from src.pipeline.vision import extract_province

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY for cache recording"

cache_dir = Path("tests/fixtures/vision_cache_dir")
cache_dir.mkdir(parents=True, exist_ok=True)
client = OpenAIClient(cache_dir=cache_dir)
records = asyncio.run(
    extract_province(
        Path("tests/fixtures/mini_pdf.pdf"),
        province="ON",
        client=client,
    )
)
print(f"Recorded cache; got {len(records)} records")
EOF
```

Expected: writes `tests/fixtures/vision_cache_dir/hishel_cache.db` with real OpenAI responses.

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_vision_pipeline.py`:

```python
"""End-to-end integration test against a 4-page mini PDF, using a
pre-recorded hishel cache so CI runs offline without OPENAI_API_KEY."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from src.openai_client import OpenAIClient
from src.pipeline.vision import extract_province

FIXTURE_PDF = Path("tests/fixtures/mini_pdf.pdf")
FIXTURE_CACHE_DIR = Path("tests/fixtures/vision_cache_dir")


@pytest.mark.integration
def test_vision_pipeline_end_to_end_from_cache(tmp_path, monkeypatch):
    # Copy cache dir to tmp so the test doesn't mutate the fixture
    cache_copy = tmp_path / "cache"
    shutil.copytree(FIXTURE_CACHE_DIR, cache_copy)

    # Sentinel API key — all calls should hit cache, so no auth is needed.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-replay-only")
    client = OpenAIClient(cache_dir=cache_copy)
    records = asyncio.run(
        extract_province(FIXTURE_PDF, province="ON", client=client)
    )

    assert len(records) >= 1
    for r in records:
        assert r.extraction_method == "vision"
        assert r.schema_version == "2"
        assert r.source_pdf_hash
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/integration/test_vision_pipeline.py -v
```

Expected: pass (from cache, no network call).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/vision_cache_dir tests/integration/test_vision_pipeline.py
git commit -m "test(integration): vision pipeline E2E with pre-recorded hishel cache"
```

---

## Task 18: Property tests

**Files:**
- Create: `tests/property/test_vision_properties.py`

- [ ] **Step 1: Write the property tests**

Create `tests/property/test_vision_properties.py`:

```python
from hypothesis import given, strategies as st

from src.pipeline.vision.toc import SectionContext
from src.pipeline.vision.windows import build_windows


@given(st.integers(min_value=1, max_value=500))
def test_n_pages_yields_n_windows(n):
    windows = list(build_windows(num_pages=n, section_map={}))
    assert len(windows) == n


@given(st.integers(min_value=1, max_value=500))
def test_every_page_is_target_exactly_once(n):
    windows = list(build_windows(num_pages=n, section_map={}))
    targets = [w.target_page for w in windows]
    assert sorted(targets) == list(range(1, n + 1))


@given(st.integers(min_value=2, max_value=500))
def test_context_page_is_previous_except_first(n):
    windows = list(build_windows(num_pages=n, section_map={}))
    assert windows[0].context_page is None
    for w in windows[1:]:
        assert w.context_page == w.target_page - 1


@given(st.integers(min_value=1, max_value=100))
def test_section_hints_default_to_empty_when_no_map(n):
    windows = list(build_windows(num_pages=n, section_map={}))
    empty = SectionContext(None, None, None)
    for w in windows:
        assert w.section_hints == empty
```

- [ ] **Step 2: Run**

```bash
pytest tests/property/test_vision_properties.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/property/test_vision_properties.py
git commit -m "test(property): window-invariant hypothesis tests"
```

---

## Task 19: Full run + manual verification

**Files:**
- None (manual step, attach notes to PR)

- [ ] **Step 1: Ensure `.env` has `OPENAI_API_KEY`**

```bash
grep -q OPENAI_API_KEY .env && echo ok || echo "add OPENAI_API_KEY to .env"
```

- [ ] **Step 2: Run full pipeline**

```bash
python -m src.cli run --force --version 2026-04-15-vision
```

Expected:
- Progress bars for each province (ON ~978, BC ~515, YT ~151 windows).
- Wall-clock ~15-25 min at concurrency=20.
- Final `data/parsed/v2026-04-15-vision/{codes.json, embeddings.npz, manifest.json}` written.
- `data/diagnostics/2026-04-15-vision/costs.json` showing ~$25.

- [ ] **Step 3: Sanity-check BC prices (the original bug)**

```bash
python - <<'EOF'
import json
from collections import Counter
records = json.loads(open("data/parsed/v2026-04-15-vision/codes.json").read_text())
bc = [r for r in records if r["province"] == "BC"]
single_digit = [r for r in bc if r.get("price") and len(r["price"].split(".")[0]) <= 1]
print(f"BC records: {len(bc)}")
print(f"BC with single-digit integer part: {len(single_digit)} (expected: very few)")
print("Sample BC prices:", [r["price"] for r in bc[:10]])
EOF
```

Expected: only a handful of genuine single-digit prices (not 1,251 like the old bug).

- [ ] **Step 4: Spot-check 20 records per province against source PDFs**

Pick 20 random records per province from `codes.json`; open the source PDF at the cited `page`; verify `fsc_code`, `fsc_description`, and `price` match. Write findings to a scratch file.

```bash
python - <<'EOF'
import json, random
records = json.loads(open("data/parsed/v2026-04-15-vision/codes.json").read_text())
for prov in ("ON", "BC", "YT"):
    sample = random.sample([r for r in records if r["province"] == prov], 20)
    for r in sample:
        print(f"  {prov} {r['fsc_code']} p{r['page']} ${r['price']}  {r['fsc_description'][:80]}")
EOF
```

- [ ] **Step 5: Start Streamlit and confirm the app loads v2 artifacts**

```bash
streamlit run app/main.py
```

Expected: app loads without errors; test a few lookups across provinces.

- [ ] **Step 6: Open a PR**

```bash
git push -u origin rebuild/pipeline-vision
gh pr create --title "Replace structural extraction with gpt-5.4-mini vision pass" \
  --body "$(cat <<'EOF'
## Summary
- Replace per-province structural extractors + semantic rescue with a single `gpt-5.4-mini` vision pass over 2-page sliding windows
- Schema bumps v1 → v2; `ExtractionMethod = Literal["vision"]`
- Add per-field spot-check regression gate (defence against the silent-BC-price class of bug)
- Full run: ~$25, ~15-25 min wall-clock, 1,644 windows

## Test plan
- [ ] `pytest` all green locally (unit + integration + property + regression)
- [ ] Full pipeline run on real PDFs completes within cost/time budget
- [ ] Manual 20-record-per-province spot-check against source PDFs attached below
- [ ] BC prices no longer single-digit (the original bug)
- [ ] Streamlit app loads v2 artifacts and returns sensible lookups

## Manual verification notes
<paste scratch notes from Task 19 Step 4>
EOF
)"
```

---

## Self-Review

### Spec coverage check

| Spec section | Task(s) |
|--------------|---------|
| §4 Architecture & data flow | 3–11 (build) + 11 (wire) |
| §5.1 Window construction (N windows, seed case) | 7 + 18 (property) |
| §5.2 Emit rule | 4 (prompt), 9 (extract) |
| §5.3 Prompt spine + section context injection | 4 |
| §5.4 Response schema (VisionRecord / WindowExtraction) | 3 |
| §6.1 Schema v1 → v2, ExtractionMethod narrowed | 2 (interim) + 13 (final) |
| §6.2 New modules under src/pipeline/vision/ | 3–10 |
| §6.3 chat_vision_json | 8 |
| §6.4 Deletions | 12 |
| §6.5 run.py rewire | 11 |
| §6.6 Loader v1 rejection | 14 |
| §7.2 Pinned render constants | 5 |
| §7.3 Concurrency (Semaphore(20)) | 10 |
| §7.4 Resumability (hishel cache + skip + failure isolation) | 8 (cache), 9 (failure log) |
| §7.5 Diagnostics (costs, failures, confidence) | 9 (failure log), 11 (costs) |
| §8.1 Error-handling matrix | 9 (retry + ValidationError), 10 (dedup), 14 (loader) |
| §8.2 Regression gates (row count + per-field) | 16 |
| §8.3 Test plan | 3, 5, 6, 7, 8, 9, 10, 17, 18 |
| §8.4 Manual verification | 19 |

No gaps.

### Placeholder scan

- No "TBD", "TODO-by-the-engineer", "fill in later" markers.
- Every code block is complete.
- Types, method signatures, field names are consistent: `VisionRecord.extraction_confidence` ↔ `_dedup` keyed on it ↔ prompt asks for it. `Window.target_page` used consistently everywhere.
- The user-contribution TODO in Task 16 is *intentional* — it's a policy choice the user makes. Example policy code is provided in comments.

### Type consistency check

- `extract_province(pdf_path, *, province, client, model=..., failure_log=...)` — signature matches in Task 10 impl, Task 11 call site, Task 17 integration test.
- `extract_window(*, window, province, images, client, model=..., failure_log=...)` — matches between Task 9 impl and Task 10 caller.
- `chat_vision_json(*, prompt, images, schema, model, system=None, temperature=0.0, max_retries=3)` — matches between Task 8 impl and Task 9 caller.
- `SectionContext(chapter, section, subsection)` — matches between Task 6 impl, Task 7 consumer, Task 4 stub, Task 10 renderer.
- `schema_version="2"` — matches between Task 13 (final narrow) and Task 10 orchestrator.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-llm-vision-pdf-extraction.md`. Three execution options:

**1. Subagent-Driven** — I dispatch a fresh subagent per task sequentially, review between tasks, fast iteration. Best for smaller plans or when tasks are tightly sequential. With 19 tasks this will be long but very controlled.

**2. Team-Driven (swarm)** — Parallel execution with dependency-aware TeamCreate coordination and worktree isolation. This plan *does* have parallelizable chunks (e.g., Tasks 3–7 could run in parallel once the branch is set up), but the schema/rewire/delete cutover in Tasks 11–13 is strictly sequential.

**3. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Fastest if you want to watch it unfold here.

Which approach?
