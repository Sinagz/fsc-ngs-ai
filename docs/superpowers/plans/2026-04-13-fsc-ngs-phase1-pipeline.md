# FSC-NGS Phase 1 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (sequential, recommended for small plans), team-driven-development (parallel swarm, recommended for 3+ tasks with parallelizable dependency graph), or superpowers:executing-plans (inline batch) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Mistral OCR pipeline with a pymupdf-first structural extractor backed by a `gpt-4o-mini` LLM rescue path, OpenAI embeddings, pydantic-validated records, and a regression gate.

**Architecture:** Two-pass extraction — a per-province structural parser emits candidate rows with explicit confidence scores; rows with confidence `< 0.8` go to a cached `gpt-4o-mini` rescue using `response_format=json_schema`. All records pass through a pydantic `FeeCodeRecord` schema that is the single source of truth. Outputs are written to versioned `data/parsed/v<DATE>/` directories with a manifest and regression diff. The Streamlit app is ported to read the new format via `src/core/matching.py` (same cosine+Jaccard algorithm as today, new location, new data contract).

**Tech Stack:** Python 3.13, pydantic v2, pymupdf, pdfplumber, openai ≥1.30, httpx + hishel (disk-cached HTTP), typer (CLI), python-dotenv, numpy. Dev: pytest, hypothesis, pytest-cov. Explicitly removed: torch, sentence-transformers, mistralai.

**Spec:** `docs/superpowers/specs/2026-04-13-fsc-ngs-rebuild-design.md`

---

## Prerequisites

- Branch `rebuild/pipeline-openai` is checked out.
- `.env` file exists at repo root with `OPENAI_API_KEY=sk-...`.
- Raw PDFs in `data/raw/pdf/` and reference DOCXs in `data/raw/docx/` (same layout as today).

---

## File Map

**New files (listed in creation order):**
- `src/__init__.py`
- `src/pipeline/__init__.py`
- `src/pipeline/schema.py` — pydantic `FeeCodeRecord`, `NGSRecord`, `Manifest`, `PageBlock`, `CandidateRow`, `Province`
- `src/pipeline/io.py` — pymupdf PDF loader, `pdf_hash()`
- `src/pipeline/structural/__init__.py`
- `src/pipeline/structural/base.py` — `StructuralExtractor` Protocol
- `src/pipeline/structural/ontario.py` — ON extractor
- `src/pipeline/structural/bc.py` — BC extractor
- `src/pipeline/structural/yukon.py` — YT extractor
- `src/pipeline/semantic.py` — LLM rescue for low-confidence rows
- `src/pipeline/validate.py` — strict pydantic gate + cross-field checks
- `src/pipeline/ngs_parser.py` — DOCX → `NGSRecord`
- `src/pipeline/ngs_mapper.py` — FSC → NGS resolution
- `src/pipeline/embed.py` — OpenAI batch embeddings
- `src/pipeline/regression.py` — diff + fail-loud gate
- `src/pipeline/run.py` — pipeline orchestrator
- `src/core/__init__.py`
- `src/core/loader.py` — load versioned artifacts
- `src/core/matching.py` — ported cosine+Jaccard search
- `src/openai_client.py` — httpx + hishel-cached OpenAI wrapper
- `src/cli.py` — typer CLI
- `tests/unit/*.py`
- `tests/integration/*.py`
- `tests/regression/*.py`
- `tests/fixtures/golden_codes.json`
- `tests/conftest.py`
- `pyproject.toml` — pytest config (we keep `requirements.txt` as the dep manifest per project convention)

**Modified:**
- `requirements.txt` — drop torch, sentence-transformers, mistralai; add openai, httpx, hishel, typer, pdfplumber, hypothesis, pytest-cov
- `app/main.py` — update imports, remove direct engine coupling
- `app/excel_export.py` — iterate pydantic fields instead of hardcoded `COLUMNS`
- `.gitignore` — add `data/diagnostics/`, `data/parsed/v*/embeddings.npz`, `.hishel-cache/`
- `CLAUDE.md` — document new pipeline
- `README.md` — update quickstart for OpenAI-based pipeline

**Deleted (last task):**
- `app/lookup_engine.py`
- `src/extract_mistral.py`
- `src/parse_mistral.py`
- `src/extract_pdfs.py`
- `src/parse_all_provinces.py`
- `src/parse_docx_full.py`
- `src/map_fsc_ngs.py`
- `src/cross_province.py`
- `src/build_embeddings.py`
- `scripts/run_pipeline.py`
- `scripts/reextract_zeros.py`

---

## Task 1: Dependency & environment setup

**Files:**
- Modify: `requirements.txt`
- Create: `.env.example`, `pyproject.toml`, `.gitignore` (modify)

- [ ] **Step 1: Update requirements.txt**

Replace the current file with:

```
# Core data pipeline
pymupdf>=1.24
pdfplumber>=0.11
python-docx>=1.1
rapidfuzz>=3.6
pydantic>=2.7
python-dotenv>=1.0

# OpenAI (extraction rescue, embeddings, reranking in Phase 2)
openai>=1.30
httpx>=0.27
hishel>=0.0.30

# CLI
typer>=0.12

# App
streamlit>=1.35
openpyxl>=3.1
pandas>=2.0
numpy>=1.26

# Dev / tests
pytest>=8.0
pytest-cov>=5.0
hypothesis>=6.0
```

- [ ] **Step 2: Create .env.example**

```
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional (defaults shown)
OPENAI_EMBED_MODEL=text-embedding-3-large
OPENAI_EMBED_DIM=1024
OPENAI_EXTRACT_MODEL=gpt-4o-mini
```

- [ ] **Step 3: Create pyproject.toml (pytest config only; requirements.txt remains dep manifest)**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra -q --strict-markers"
markers = [
    "unit: fast isolated tests",
    "integration: hit local files / small PDF fixtures",
    "regression: snapshot-based, fails on data drift",
    "property: hypothesis-based property tests",
]

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
show_missing = true
skip_empty = true
```

- [ ] **Step 4: Update .gitignore**

Append to the existing `.gitignore`:

```
# Phase 1 additions
data/diagnostics/
data/parsed/v*/embeddings.npz
.hishel-cache/
.pytest_cache/
htmlcov/
.coverage
```

- [ ] **Step 5: Install deps and commit**

```bash
pip install -r requirements.txt
git add requirements.txt .env.example pyproject.toml .gitignore
git commit -m "chore: pin new pipeline deps, drop torch/mistral"
```

Expected: `pip install` succeeds. No ImportError when later tasks import `openai`, `httpx`, `hishel`, `typer`, `pdfplumber`, `hypothesis`.

---

## Task 2: Schema module

**Files:**
- Create: `src/__init__.py`, `src/pipeline/__init__.py`, `src/pipeline/schema.py`
- Test: `tests/unit/test_schema.py`, `tests/conftest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Create `tests/unit/test_schema.py`:

```python
from decimal import Decimal
import pytest
from pydantic import ValidationError

from src.pipeline.schema import (
    FeeCodeRecord, NGSRecord, Manifest, CandidateRow, PageBlock,
)


def _valid_record_kwargs() -> dict:
    return dict(
        province="ON", fsc_code="K040", fsc_fn="Periodic health visit",
        fsc_description="General periodic visit, adult.", page=47,
        source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
    )


def test_record_roundtrips_through_json():
    r = FeeCodeRecord(**_valid_record_kwargs(), price=Decimal("78.45"))
    data = r.model_dump(mode="json")
    assert FeeCodeRecord(**data) == r


def test_record_is_frozen():
    r = FeeCodeRecord(**_valid_record_kwargs())
    with pytest.raises(ValidationError):
        r.province = "BC"  # type: ignore[misc]


def test_record_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        FeeCodeRecord(**_valid_record_kwargs(), nope="x")


def test_record_rejects_unknown_province():
    kwargs = _valid_record_kwargs() | {"province": "AB"}
    with pytest.raises(ValidationError):
        FeeCodeRecord(**kwargs)


def test_record_rejects_negative_confidence():
    kwargs = _valid_record_kwargs() | {"extraction_confidence": -0.1}
    with pytest.raises(ValidationError):
        FeeCodeRecord(**kwargs)


def test_candidate_row_is_frozen():
    row = CandidateRow(
        province="ON", fsc_code="K040", fsc_fn="", fsc_description="",
        page=1, confidence=0.9, source_pdf_hash="a" * 64,
    )
    with pytest.raises(ValidationError):
        row.confidence = 0.1  # type: ignore[misc]


def test_manifest_records_schema_version():
    m = Manifest(
        schema_version="1", generated_at="2026-04-13T00:00:00Z",
        git_sha="deadbeef", row_counts={"ON": 0, "BC": 0, "YT": 0},
        source_pdf_hashes={}, models={"embed": "text-embedding-3-large"},
    )
    assert m.schema_version == "1"


def test_ngs_record_roundtrips():
    n = NGSRecord(ngs_code="1AA", ngs_label="Health exam",
                  ngs_description="routine visit", code_refs=["K040"])
    assert NGSRecord(**n.model_dump()) == n
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.pipeline.schema'`

- [ ] **Step 3: Create the module**

`src/__init__.py`: empty file.

`src/pipeline/__init__.py`: empty file.

`src/pipeline/schema.py`:

```python
"""Canonical pydantic schemas. Single source of truth for the pipeline."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Province = Literal["ON", "BC", "YT"]
ExtractionMethod = Literal["structural", "semantic"]
NGSMethod = Literal["exact", "llm", "manual"]


class PageBlock(BaseModel):
    """One pymupdf text block with layout metadata."""
    model_config = ConfigDict(frozen=True)

    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font: str
    size: float


class CandidateRow(BaseModel):
    """Structural-parser output before validation + NGS mapping."""
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
    source_pdf_hash: str
    confidence: float = Field(ge=0.0, le=1.0)


class FeeCodeRecord(BaseModel):
    """Validated, NGS-mapped fee code record. The canonical type."""
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
    source_pdf_hash: str
    extraction_method: ExtractionMethod
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    NGS_code: str | None = None
    NGS_label: str | None = None
    NGS_mapping_method: NGSMethod | None = None
    NGS_mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class NGSRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ngs_code: str
    ngs_label: str
    ngs_description: str = ""
    code_refs: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    generated_at: str
    git_sha: str
    row_counts: dict[str, int]
    source_pdf_hashes: dict[str, str]
    models: dict[str, str]
    regression_override: str | None = None
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_schema.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ tests/conftest.py tests/unit/test_schema.py
git commit -m "feat(pipeline): add pydantic schema module"
```

---

## Task 3: OpenAI client wrapper

**Files:**
- Create: `src/openai_client.py`
- Test: `tests/unit/test_openai_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_openai_client.py
from unittest.mock import MagicMock, patch
import pytest

from src.openai_client import OpenAIClient, CostTracker


def test_cost_tracker_accumulates():
    t = CostTracker()
    t.record(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    t.record(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=0)
    snap = t.snapshot()
    assert snap["gpt-4o-mini"]["prompt_tokens"] == 300
    assert snap["gpt-4o-mini"]["completion_tokens"] == 50


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_chat_json_validates_against_schema(monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        code: str
        price: float

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '{"code": "K040", "price": 78.45}'
    fake_resp.usage.prompt_tokens = 50
    fake_resp.usage.completion_tokens = 10
    fake_resp.model = "gpt-4o-mini"

    with patch.object(client._sdk.chat.completions, "create", return_value=fake_resp):
        parsed = client.chat_json(prompt="hi", schema=Out, model="gpt-4o-mini")

    assert parsed == Out(code="K040", price=78.45)
    assert client.costs.snapshot()["gpt-4o-mini"]["prompt_tokens"] == 50


def test_embed_batches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    def fake_create(input, model, dimensions, **kw):
        return MagicMock(
            data=[MagicMock(embedding=[0.1] * dimensions) for _ in input],
            usage=MagicMock(prompt_tokens=len(input), total_tokens=len(input)),
            model=model,
        )

    with patch.object(client._sdk.embeddings, "create", side_effect=fake_create):
        vecs = client.embed(["a"] * 1200, model="text-embedding-3-large", dim=1024)

    assert len(vecs) == 1200
    assert len(vecs[0]) == 1024
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_openai_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.openai_client'`

- [ ] **Step 3: Create the wrapper**

`src/openai_client.py`:

```python
"""Single chokepoint for all OpenAI calls.
No module in src/pipeline or src/core imports openai directly."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
import hishel
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class CostTracker:
    _by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, *, model: str, prompt_tokens: int, completion_tokens: int = 0) -> None:
        slot = self._by_model.setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        slot["prompt_tokens"] += prompt_tokens
        slot["completion_tokens"] += completion_tokens
        slot["calls"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {m: dict(s) for m, s in self._by_model.items()}


class OpenAIClient:
    """Cached, retrying wrapper around the OpenAI SDK."""

    def __init__(self, cache_dir: str = ".hishel-cache") -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment")

        storage = hishel.FileStorage(base_path=cache_dir)
        transport = hishel.CacheTransport(
            transport=httpx.HTTPTransport(), storage=storage
        )
        http_client = httpx.Client(transport=transport, timeout=60.0)

        self._sdk = OpenAI(api_key=key, http_client=http_client)
        self.costs = CostTracker()

    def chat_json(
        self,
        *,
        prompt: str,
        schema: type[T],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> T:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        json_schema = {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        }

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._sdk.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_schema", "json_schema": json_schema},
                )
                self.costs.record(
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
                return schema.model_validate(json.loads(resp.choices[0].message.content))
            except Exception as exc:
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"chat_json failed after {max_retries} retries: {last_err}")

    def embed(
        self, texts: list[str], *, model: str, dim: int, batch_size: int = 500
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._sdk.embeddings.create(
                input=batch, model=model, dimensions=dim
            )
            self.costs.record(
                model=resp.model,
                prompt_tokens=resp.usage.prompt_tokens,
            )
            out.extend(d.embedding for d in resp.data)
        return out
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_openai_client.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/openai_client.py tests/unit/test_openai_client.py
git commit -m "feat: add httpx+hishel-cached OpenAI client wrapper"
```

---

## Task 4: PDF IO module

**Files:**
- Create: `src/pipeline/io.py`
- Test: `tests/unit/test_io.py`, `tests/fixtures/tiny.pdf` (generated at test time)

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_io.py`:

```python
import hashlib
import fitz  # pymupdf

from src.pipeline.io import load_pdf, pdf_hash


def _make_tiny_pdf(path, text="K040 Periodic visit 78.45"):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_load_pdf_returns_page_blocks(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_tiny_pdf(pdf)
    blocks = load_pdf(pdf)
    assert len(blocks) >= 1
    assert blocks[0].page == 1
    assert "K040" in blocks[0].text


def test_pdf_hash_is_stable(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_tiny_pdf(pdf)
    h1 = pdf_hash(pdf)
    h2 = pdf_hash(pdf)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_pdf_hash_differs_for_different_content(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    _make_tiny_pdf(a, text="K040")
    _make_tiny_pdf(b, text="K041")
    assert pdf_hash(a) != pdf_hash(b)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_io.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/pipeline/io.py`:

```python
"""PDF loading via pymupdf, returning structured PageBlock records."""
from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # pymupdf

from src.pipeline.schema import PageBlock


def pdf_hash(path: Path) -> str:
    """SHA-256 of the raw PDF bytes. Stable across runs, changes when PDF changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pdf(path: Path) -> list[PageBlock]:
    """Load all text blocks from a PDF with layout and font metadata."""
    blocks: list[PageBlock] = []
    doc = fitz.open(str(path))
    try:
        for page_idx, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = (span.get("text") or "").strip()
                        if not text:
                            continue
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        blocks.append(PageBlock(
                            page=page_idx, text=text,
                            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                            font=span.get("font", ""), size=float(span.get("size", 0.0)),
                        ))
    finally:
        doc.close()
    return blocks
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_io.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/io.py tests/unit/test_io.py
git commit -m "feat(pipeline): add pymupdf-based PDF loader"
```

---

## Task 5: Structural base (Protocol + confidence rules)

**Files:**
- Create: `src/pipeline/structural/__init__.py`, `src/pipeline/structural/base.py`
- Test: `tests/unit/test_structural_base.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_structural_base.py`:

```python
from src.pipeline.structural.base import (
    Confidence, group_by_page, detect_section_headers, is_section_header,
)
from src.pipeline.schema import PageBlock


def _blk(page, text, size=10.0, y=100.0):
    return PageBlock(page=page, text=text, x0=72, y0=y, x1=100, y1=y+10,
                     font="Helvetica", size=size)


def test_confidence_constants_cover_threshold():
    assert Confidence.HIGH == 1.0
    assert Confidence.ADJACENT_FIELD == 0.85
    assert Confidence.AMBIGUOUS == 0.3
    # Threshold for LLM rescue is 0.8 per spec
    assert Confidence.ADJACENT_FIELD >= 0.8
    assert Confidence.MISSING_FIELD < 0.8


def test_group_by_page_preserves_order():
    blocks = [
        _blk(1, "A", y=100), _blk(2, "B", y=200),
        _blk(1, "C", y=150), _blk(3, "D", y=50),
    ]
    grouped = group_by_page(blocks)
    assert list(grouped.keys()) == [1, 2, 3]
    # within a page, sorted by y then x
    assert [b.text for b in grouped[1]] == ["A", "C"]


def test_is_section_header_detects_large_font():
    assert is_section_header(_blk(1, "ANAESTHESIA", size=14.0), body_size=10.0)
    assert not is_section_header(_blk(1, "K040", size=10.0), body_size=10.0)


def test_detect_section_headers_emits_pairs():
    blocks = [
        _blk(1, "CHAPTER A", size=14, y=50),
        _blk(1, "K040 Foo 78.45", size=10, y=80),
        _blk(2, "CHAPTER B", size=14, y=50),
        _blk(2, "K041 Bar 99.00", size=10, y=80),
    ]
    headers = detect_section_headers(blocks, header_size=14.0)
    assert headers == [(1, "CHAPTER A"), (2, "CHAPTER B")]
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_structural_base.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/structural/__init__.py`: empty file.

`src/pipeline/structural/base.py`:

```python
"""Shared utilities and Protocol for per-province structural extractors."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import ClassVar, Iterator, Protocol

from src.pipeline.schema import CandidateRow, PageBlock, Province


class Confidence:
    """Explicit confidence bands. Spec: rescue threshold is 0.8."""
    HIGH = 1.0               # all fields matched cleanly
    ADJACENT_FIELD = 0.85    # one field recovered from adjacent line
    MISSING_FIELD = 0.6      # matched code regex, missing price or description
    AMBIGUOUS = 0.3          # possibly a table header
    REJECT = 0.0


class StructuralExtractor(Protocol):
    PROVINCE: ClassVar[Province]
    CODE_REGEX: ClassVar[re.Pattern[str]]

    def extract(
        self, pages: list[PageBlock], source_pdf_hash: str
    ) -> Iterator[CandidateRow]: ...


def group_by_page(blocks: list[PageBlock]) -> dict[int, list[PageBlock]]:
    grouped: dict[int, list[PageBlock]] = defaultdict(list)
    for b in blocks:
        grouped[b.page].append(b)
    for page in grouped:
        grouped[page].sort(key=lambda b: (b.y0, b.x0))
    return dict(sorted(grouped.items()))


def is_section_header(block: PageBlock, *, body_size: float, tolerance: float = 0.5) -> bool:
    return block.size >= body_size + 1.5 - tolerance and len(block.text) >= 3


def detect_section_headers(
    blocks: list[PageBlock], *, header_size: float, tolerance: float = 0.3
) -> list[tuple[int, str]]:
    return [
        (b.page, b.text)
        for b in blocks
        if abs(b.size - header_size) <= tolerance
    ]
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_structural_base.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/structural/ tests/unit/test_structural_base.py
git commit -m "feat(pipeline): add structural extractor protocol and helpers"
```

---

## Task 6: Ontario structural extractor

**Files:**
- Create: `src/pipeline/structural/ontario.py`
- Test: `tests/unit/test_structural_ontario.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_structural_ontario.py
from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.ontario import OntarioExtractor, ON_CODE_REGEX


def _blk(page, text, size=10.0, y=100.0, x=72.0):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x+50, y1=y+10,
                     font="Helvetica", size=size)


def test_code_regex_matches_on_patterns():
    for code in ["K040", "A007", "Z432", "K9999"]:
        assert ON_CODE_REGEX.fullmatch(code)
    for bad in ["01712", "0615", "KK40", "1234"]:
        assert not ON_CODE_REGEX.fullmatch(bad)


def test_extract_simple_row_high_confidence():
    pages = [
        _blk(1, "GENERAL PREAMBLE", size=14, y=40),
        _blk(1, "K040", size=10, y=100, x=72),
        _blk(1, "Periodic health visit", size=10, y=100, x=140),
        _blk(1, "78.45", size=10, y=100, x=500),
    ]
    ex = OntarioExtractor()
    rows = list(ex.extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    row = rows[0]
    assert row.fsc_code == "K040"
    assert "Periodic" in row.fsc_description
    assert row.price == Decimal("78.45")
    assert row.fsc_chapter == "GENERAL PREAMBLE"
    assert row.confidence >= 0.8
    assert row.province == "ON"


def test_extract_missing_price_lower_confidence():
    pages = [
        _blk(1, "CHAPTER", size=14, y=40),
        _blk(1, "K040", size=10, y=100),
        _blk(1, "Periodic visit", size=10, y=100, x=140),
    ]
    rows = list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64))
    assert len(rows) == 1
    assert rows[0].confidence < 0.8
    assert rows[0].price is None


def test_extract_skips_non_code_rows():
    pages = [
        _blk(1, "CHAPTER", size=14, y=40),
        _blk(1, "See notes.", size=10, y=100),
    ]
    assert list(OntarioExtractor().extract(pages, source_pdf_hash="a" * 64)) == []
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_structural_ontario.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/structural/ontario.py`:

```python
"""Ontario MOH Schedule of Benefits — structural extractor.

Rules verified against the 2024-12 edition. If the publisher ships a new
edition, re-check CHAPTER_FONT_SIZE, BODY_FONT_SIZE, and the column x-ranges
before trusting the output.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import ClassVar, Iterator

from src.pipeline.schema import CandidateRow, PageBlock, Province
from src.pipeline.structural.base import (
    Confidence, StructuralExtractor, group_by_page,
)

ON_CODE_REGEX = re.compile(r"[A-Z]\d{3,4}")

CHAPTER_FONT_SIZE = 14.0
SECTION_FONT_SIZE = 11.5
BODY_FONT_SIZE = 10.0
FONT_TOLERANCE = 0.5

CODE_COLUMN_X = 72.0
CODE_COLUMN_TOLERANCE = 6.0

# y-tolerance: spans on the "same row" share y0 within this many points
ROW_Y_TOLERANCE = 2.5


class OntarioExtractor:
    PROVINCE: ClassVar[Province] = "ON"
    CODE_REGEX: ClassVar[re.Pattern[str]] = ON_CODE_REGEX

    def extract(
        self, pages: list[PageBlock], source_pdf_hash: str
    ) -> Iterator[CandidateRow]:
        grouped = group_by_page(pages)
        chapter: str | None = None
        section: str | None = None

        for page_num, blocks in grouped.items():
            for block in blocks:
                if self._is_chapter(block):
                    chapter = block.text.strip()
                    section = None
                    continue
                if self._is_section(block):
                    section = block.text.strip()
                    continue
                # skip non-body
                if abs(block.size - BODY_FONT_SIZE) > FONT_TOLERANCE:
                    continue

            row_groups = self._group_into_rows(blocks)
            for row_blocks in row_groups:
                cr = self._try_row(
                    row_blocks, page_num, chapter, section, source_pdf_hash
                )
                if cr is not None:
                    yield cr

    def _is_chapter(self, b: PageBlock) -> bool:
        return abs(b.size - CHAPTER_FONT_SIZE) <= FONT_TOLERANCE and b.text.isupper()

    def _is_section(self, b: PageBlock) -> bool:
        return abs(b.size - SECTION_FONT_SIZE) <= FONT_TOLERANCE

    def _group_into_rows(self, blocks: list[PageBlock]) -> list[list[PageBlock]]:
        rows: list[list[PageBlock]] = []
        current: list[PageBlock] = []
        current_y: float | None = None
        for b in blocks:
            if abs(b.size - BODY_FONT_SIZE) > FONT_TOLERANCE:
                continue
            if current_y is None or abs(b.y0 - current_y) <= ROW_Y_TOLERANCE:
                current.append(b)
                current_y = b.y0 if current_y is None else current_y
            else:
                rows.append(current)
                current = [b]
                current_y = b.y0
        if current:
            rows.append(current)
        return rows

    def _try_row(
        self,
        blocks: list[PageBlock],
        page_num: int,
        chapter: str | None,
        section: str | None,
        source_pdf_hash: str,
    ) -> CandidateRow | None:
        code_block = next(
            (b for b in blocks
             if self.CODE_REGEX.fullmatch(b.text)
             and abs(b.x0 - CODE_COLUMN_X) <= CODE_COLUMN_TOLERANCE),
            None,
        )
        if code_block is None:
            return None

        other = [b for b in blocks if b is not code_block]
        description_blocks = [b for b in other if b.x0 > CODE_COLUMN_X + CODE_COLUMN_TOLERANCE]
        description_blocks.sort(key=lambda b: b.x0)

        price: Decimal | None = None
        desc_parts: list[str] = []
        for b in description_blocks:
            if self._looks_like_price(b.text):
                try:
                    price = Decimal(b.text.replace(",", "").replace("$", ""))
                    continue
                except InvalidOperation:
                    pass
            desc_parts.append(b.text)

        description = " ".join(desc_parts).strip()
        fsc_fn = description.split(".")[0][:80] if description else ""

        if description and price is not None:
            confidence = Confidence.HIGH
        elif description or price is not None:
            confidence = Confidence.MISSING_FIELD
        else:
            confidence = Confidence.AMBIGUOUS

        return CandidateRow(
            province="ON",
            fsc_code=code_block.text,
            fsc_fn=fsc_fn,
            fsc_description=description,
            fsc_chapter=chapter,
            fsc_section=section,
            price=price,
            page=page_num,
            source_pdf_hash=source_pdf_hash,
            confidence=confidence,
        )

    @staticmethod
    def _looks_like_price(text: str) -> bool:
        cleaned = text.replace(",", "").replace("$", "").strip()
        if not cleaned:
            return False
        try:
            Decimal(cleaned)
            return "." in cleaned
        except InvalidOperation:
            return False
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_structural_ontario.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/structural/ontario.py tests/unit/test_structural_ontario.py
git commit -m "feat(pipeline): add Ontario structural extractor"
```

---

## Task 7: BC structural extractor

**Files:**
- Create: `src/pipeline/structural/bc.py`
- Test: `tests/unit/test_structural_bc.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_structural_bc.py
from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.bc import BCExtractor, BC_CODE_REGEX


def _blk(page, text, size=10.0, y=100.0, x=72.0):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x+50, y1=y+10,
                     font="Helvetica", size=size)


def test_code_regex_bc_five_digits():
    assert BC_CODE_REGEX.fullmatch("01712")
    assert BC_CODE_REGEX.fullmatch("99999")
    assert not BC_CODE_REGEX.fullmatch("K040")
    assert not BC_CODE_REGEX.fullmatch("9999")
    assert not BC_CODE_REGEX.fullmatch("012345")


def test_extract_bc_row():
    pages = [
        _blk(1, "CONSULTATIONS", size=13, y=40),
        _blk(1, "01712", size=10, y=100, x=72),
        _blk(1, "Limited consultation", size=10, y=100, x=140),
        _blk(1, "45.20", size=10, y=100, x=500),
    ]
    rows = list(BCExtractor().extract(pages, source_pdf_hash="b" * 64))
    assert len(rows) == 1
    r = rows[0]
    assert r.province == "BC"
    assert r.fsc_code == "01712"
    assert r.price == Decimal("45.20")
    assert r.confidence >= 0.8
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_structural_bc.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/structural/bc.py`:

```python
"""BC MSC Payment Schedule — structural extractor.
Verified against the 2024-2025 edition."""
from __future__ import annotations

import re
from typing import ClassVar

from src.pipeline.schema import Province
from src.pipeline.structural.base import Confidence
from src.pipeline.structural.ontario import OntarioExtractor  # reuse row parsing

BC_CODE_REGEX = re.compile(r"\d{5}")

# BC uses slightly different font sizes
BC_CHAPTER_FONT = 13.0
BC_BODY_FONT = 10.0


class BCExtractor(OntarioExtractor):
    """BC layout is structurally similar to ON — inherit row parsing,
    override the code regex and province constant."""
    PROVINCE: ClassVar[Province] = "BC"
    CODE_REGEX: ClassVar[re.Pattern[str]] = BC_CODE_REGEX

    def _is_chapter(self, b):
        from src.pipeline.structural.ontario import FONT_TOLERANCE
        return abs(b.size - BC_CHAPTER_FONT) <= FONT_TOLERANCE and b.text.isupper()

    def _try_row(self, blocks, page_num, chapter, section, source_pdf_hash):
        row = super()._try_row(blocks, page_num, chapter, section, source_pdf_hash)
        if row is None:
            return None
        # overwrite province field
        return row.model_copy(update={"province": "BC"})
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_structural_bc.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/structural/bc.py tests/unit/test_structural_bc.py
git commit -m "feat(pipeline): add BC structural extractor"
```

---

## Task 8: Yukon structural extractor

**Files:**
- Create: `src/pipeline/structural/yukon.py`
- Test: `tests/unit/test_structural_yukon.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_structural_yukon.py
from decimal import Decimal

from src.pipeline.schema import PageBlock
from src.pipeline.structural.yukon import YukonExtractor, YT_CODE_REGEX


def _blk(page, text, size=10.0, y=100.0, x=72.0):
    return PageBlock(page=page, text=text, x0=x, y0=y, x1=x+50, y1=y+10,
                     font="Helvetica", size=size)


def test_code_regex_yt_four_digits():
    assert YT_CODE_REGEX.fullmatch("0615")
    assert YT_CODE_REGEX.fullmatch("9999")
    assert not YT_CODE_REGEX.fullmatch("01712")
    assert not YT_CODE_REGEX.fullmatch("K040")


def test_extract_yt_row():
    pages = [
        _blk(1, "GENERAL PRACTICE", size=13, y=40),
        _blk(1, "0615", size=10, y=100, x=72),
        _blk(1, "Office visit", size=10, y=100, x=140),
        _blk(1, "55.00", size=10, y=100, x=500),
    ]
    rows = list(YukonExtractor().extract(pages, source_pdf_hash="c" * 64))
    assert len(rows) == 1
    assert rows[0].province == "YT"
    assert rows[0].fsc_code == "0615"
    assert rows[0].price == Decimal("55.00")
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_structural_yukon.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/structural/yukon.py`:

```python
"""Yukon Physician Fee Guide — structural extractor.
Verified against the 2024 edition."""
from __future__ import annotations

import re
from typing import ClassVar

from src.pipeline.schema import Province
from src.pipeline.structural.bc import BCExtractor

YT_CODE_REGEX = re.compile(r"\d{4}")


class YukonExtractor(BCExtractor):
    """Yukon layout mirrors BC closely — inherit everything except code regex."""
    PROVINCE: ClassVar[Province] = "YT"
    CODE_REGEX: ClassVar[re.Pattern[str]] = YT_CODE_REGEX

    def _try_row(self, blocks, page_num, chapter, section, source_pdf_hash):
        # Bypass BC's override chain to set province correctly
        from src.pipeline.structural.ontario import OntarioExtractor
        row = OntarioExtractor._try_row(
            self, blocks, page_num, chapter, section, source_pdf_hash
        )
        if row is None:
            return None
        return row.model_copy(update={"province": "YT"})
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_structural_yukon.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/structural/yukon.py tests/unit/test_structural_yukon.py
git commit -m "feat(pipeline): add Yukon structural extractor"
```

---

## Task 9: Semantic (LLM) rescue

**Files:**
- Create: `src/pipeline/semantic.py`
- Test: `tests/unit/test_semantic.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_semantic.py
from decimal import Decimal
from unittest.mock import MagicMock

from src.pipeline.schema import CandidateRow
from src.pipeline.semantic import rescue, RescueOutput


def _low_conf_row():
    return CandidateRow(
        province="ON", fsc_code="K040", fsc_fn="", fsc_description="",
        page=47, source_pdf_hash="a" * 64, confidence=0.6,
    )


def test_rescue_skips_rows_above_threshold():
    row = _low_conf_row().model_copy(update={"confidence": 0.9})
    client = MagicMock()
    rescued, unresolved = rescue([row], client=client, model="gpt-4o-mini",
                                 context_lines={}, threshold=0.8)
    assert rescued == [row]
    assert unresolved == []
    client.chat_json.assert_not_called()


def test_rescue_calls_llm_for_low_confidence():
    row = _low_conf_row()
    client = MagicMock()
    client.chat_json.return_value = RescueOutput(
        fsc_fn="Periodic visit", fsc_description="General periodic visit, adult.",
        price="78.45", confidence=0.9, resolved=True,
    )
    rescued, unresolved = rescue([row], client=client, model="gpt-4o-mini",
                                 context_lines={(47, "K040"): "surrounding text"},
                                 threshold=0.8)
    assert len(rescued) == 1
    assert rescued[0].fsc_description == "General periodic visit, adult."
    assert rescued[0].price == Decimal("78.45")
    assert unresolved == []


def test_rescue_logs_unresolved():
    row = _low_conf_row()
    client = MagicMock()
    client.chat_json.return_value = RescueOutput(
        fsc_fn="", fsc_description="", price=None, confidence=0.0, resolved=False,
    )
    rescued, unresolved = rescue([row], client=client, model="gpt-4o-mini",
                                 context_lines={(47, "K040"): ""}, threshold=0.8)
    assert rescued == []
    assert unresolved == [row]
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_semantic.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/semantic.py`:

```python
"""LLM rescue path for low-confidence structural rows.
Only rows with confidence < threshold (default 0.8) are sent to the LLM."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Protocol

from pydantic import BaseModel, Field

from src.pipeline.schema import CandidateRow


class RescueOutput(BaseModel):
    fsc_fn: str = ""
    fsc_description: str = ""
    price: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    resolved: bool


class _ClientLike(Protocol):
    def chat_json(self, *, prompt: str, schema: type, model: str,
                  system: str | None = None, temperature: float = 0.0) -> BaseModel: ...


SYSTEM_PROMPT = (
    "You extract Canadian physician fee codes from PDF text. "
    "Return strictly structured JSON matching the schema. "
    "If you cannot confidently extract the fields, set resolved=false."
)

USER_TEMPLATE = """Province: {province}
Code: {code}
Page: {page}

Surrounding text:
---
{context}
---

Extract fsc_fn (short name), fsc_description (full text), and price (decimal number)."""


def rescue(
    rows: list[CandidateRow],
    *,
    client: _ClientLike,
    model: str,
    context_lines: dict[tuple[int, str], str],
    threshold: float = 0.8,
) -> tuple[list[CandidateRow], list[CandidateRow]]:
    """Return (rescued_rows, unresolved_rows).
    Rows with confidence >= threshold pass through unchanged."""
    rescued: list[CandidateRow] = []
    unresolved: list[CandidateRow] = []
    for row in rows:
        if row.confidence >= threshold:
            rescued.append(row)
            continue

        context = context_lines.get((row.page, row.fsc_code), "")
        prompt = USER_TEMPLATE.format(
            province=row.province, code=row.fsc_code, page=row.page, context=context
        )
        out = client.chat_json(
            prompt=prompt, schema=RescueOutput, model=model,
            system=SYSTEM_PROMPT, temperature=0.0,
        )
        if not out.resolved:
            unresolved.append(row)
            continue

        price: Decimal | None = None
        if out.price:
            try:
                price = Decimal(out.price.replace("$", "").replace(",", ""))
            except InvalidOperation:
                price = None

        rescued.append(row.model_copy(update={
            "fsc_fn": out.fsc_fn,
            "fsc_description": out.fsc_description,
            "price": price,
            "confidence": max(row.confidence, out.confidence),
        }))
    return rescued, unresolved
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_semantic.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/semantic.py tests/unit/test_semantic.py
git commit -m "feat(pipeline): add LLM rescue for low-confidence rows"
```

---

## Task 10: Validator

**Files:**
- Create: `src/pipeline/validate.py`
- Test: `tests/unit/test_validate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_validate.py
from decimal import Decimal

from src.pipeline.schema import CandidateRow
from src.pipeline.validate import validate, ValidationError


def _row(province="ON", code="K040", desc="desc", price="78.45", conf=1.0):
    return CandidateRow(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description=desc,
        page=1, source_pdf_hash="a" * 64, confidence=conf,
        price=Decimal(price) if price else None,
    )


def test_validate_accepts_clean_row():
    rows, rejects = validate([_row()])
    assert len(rows) == 1
    assert rejects == []
    assert rows[0].fsc_code == "K040"


def test_validate_rejects_wrong_province_regex():
    rows, rejects = validate([_row(province="ON", code="01712")])
    assert rows == []
    assert len(rejects) == 1
    assert "regex" in rejects[0].reason.lower()


def test_validate_rejects_duplicate_within_province():
    rows, rejects = validate([_row(), _row()])
    assert len(rows) == 1
    assert len(rejects) == 1
    assert "duplicate" in rejects[0].reason.lower()


def test_validate_allows_same_code_across_provinces():
    rows, rejects = validate([_row(province="ON", code="K040"),
                              _row(province="BC", code="01712")])
    assert len(rows) == 2
    assert rejects == []
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_validate.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/validate.py`:

```python
"""Strict gate between CandidateRow and FeeCodeRecord.
Emits per-reject reasons for diagnostics."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from src.pipeline.schema import CandidateRow, FeeCodeRecord, Province

PROVINCE_REGEX: dict[Province, Pattern[str]] = {
    "ON": re.compile(r"[A-Z]\d{3,4}"),
    "BC": re.compile(r"\d{5}"),
    "YT": re.compile(r"\d{4}"),
}


@dataclass(frozen=True)
class ValidationError:
    candidate: CandidateRow
    reason: str


def validate(
    rows: list[CandidateRow],
) -> tuple[list[FeeCodeRecord], list[ValidationError]]:
    seen: dict[tuple[Province, str], CandidateRow] = {}
    records: list[FeeCodeRecord] = []
    rejects: list[ValidationError] = []

    for row in rows:
        pattern = PROVINCE_REGEX[row.province]
        if not pattern.fullmatch(row.fsc_code):
            rejects.append(ValidationError(row, f"code regex mismatch for {row.province}"))
            continue

        key = (row.province, row.fsc_code)
        if key in seen:
            rejects.append(ValidationError(
                row, f"duplicate {row.province}:{row.fsc_code} (first seen page {seen[key].page})"
            ))
            continue
        seen[key] = row

        if row.price is not None and row.price <= 0:
            rejects.append(ValidationError(row, "price must be positive"))
            continue

        record = FeeCodeRecord(
            province=row.province,
            fsc_code=row.fsc_code,
            fsc_fn=row.fsc_fn,
            fsc_description=row.fsc_description,
            fsc_chapter=row.fsc_chapter,
            fsc_section=row.fsc_section,
            fsc_subsection=row.fsc_subsection,
            fsc_notes=row.fsc_notes,
            price=row.price,
            page=row.page,
            source_pdf_hash=row.source_pdf_hash,
            extraction_method="structural" if row.confidence >= 0.8 else "semantic",
            extraction_confidence=row.confidence,
        )
        records.append(record)

    return records, rejects
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_validate.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/validate.py tests/unit/test_validate.py
git commit -m "feat(pipeline): add strict validator with per-reject reasons"
```

---

## Task 11: NGS DOCX parser

**Files:**
- Create: `src/pipeline/ngs_parser.py`
- Test: `tests/unit/test_ngs_parser.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_ngs_parser.py
from pathlib import Path
import docx

from src.pipeline.ngs_parser import parse_ngs_docx


def _make_docx(tmp_path: Path) -> Path:
    doc = docx.Document()
    doc.add_paragraph("1AA Health exam", style="Heading 1")
    doc.add_paragraph("Routine periodic visit for an adult patient.")
    doc.add_paragraph("Includes codes: K040, K005")
    doc.add_paragraph("1AB Consultation", style="Heading 1")
    doc.add_paragraph("Specialist consultation.")
    doc.add_paragraph("Includes codes: A005, 01712")
    out = tmp_path / "ngs.docx"
    doc.save(str(out))
    return out


def test_parse_ngs_docx_emits_records(tmp_path):
    path = _make_docx(tmp_path)
    records = parse_ngs_docx(path)
    assert len(records) == 2
    assert records[0].ngs_code == "1AA"
    assert records[0].ngs_label == "Health exam"
    assert "K040" in records[0].code_refs
    assert "01712" in records[1].code_refs
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_ngs_parser.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/ngs_parser.py`:

```python
"""Parse CIHI NGS reference DOCX files into NGSRecord objects."""
from __future__ import annotations

import re
from pathlib import Path

import docx

from src.pipeline.schema import NGSRecord

NGS_HEADING_RE = re.compile(r"^(\S+)\s+(.+)$")
CODE_LIST_RE = re.compile(r"[A-Z]?\d{3,5}")


def parse_ngs_docx(path: Path) -> list[NGSRecord]:
    """Heading-1 paragraphs are NGS headers ("<code> <label>").
    Body paragraphs between headers form the description; any
    "Includes codes:" line contributes to code_refs."""
    doc = docx.Document(str(path))
    records: list[NGSRecord] = []
    current_code: str | None = None
    current_label = ""
    desc_parts: list[str] = []
    code_refs: list[str] = []

    def _flush() -> None:
        nonlocal current_code, current_label, desc_parts, code_refs
        if current_code is not None:
            records.append(NGSRecord(
                ngs_code=current_code,
                ngs_label=current_label,
                ngs_description=" ".join(desc_parts).strip(),
                code_refs=list(dict.fromkeys(code_refs)),
            ))
        current_code = None
        current_label = ""
        desc_parts = []
        code_refs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            _flush()
            m = NGS_HEADING_RE.match(text)
            if m:
                current_code = m.group(1)
                current_label = m.group(2)
            continue
        if text.lower().startswith("includes codes"):
            code_refs.extend(CODE_LIST_RE.findall(text))
        else:
            desc_parts.append(text)
    _flush()
    return records
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_ngs_parser.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ngs_parser.py tests/unit/test_ngs_parser.py
git commit -m "feat(pipeline): add NGS DOCX parser"
```

---

## Task 12: NGS mapper

**Files:**
- Create: `src/pipeline/ngs_mapper.py`
- Test: `tests/unit/test_ngs_mapper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_ngs_mapper.py
from decimal import Decimal
from unittest.mock import MagicMock

from src.pipeline.schema import FeeCodeRecord, NGSRecord
from src.pipeline.ngs_mapper import map_ngs, NGSVerdict


def _fee(code="K040"):
    return FeeCodeRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="Periodic visit",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
    )


def _ngs(code="1AA", refs=("K040",)):
    return NGSRecord(ngs_code=code, ngs_label="Exam",
                     ngs_description="Periodic visit", code_refs=list(refs))


def test_exact_match_sets_method_exact():
    client = MagicMock()
    records = map_ngs([_fee()], [_ngs(refs=("K040",))], client=client,
                      embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code == "1AA"
    assert records[0].NGS_mapping_method == "exact"
    assert records[0].NGS_mapping_confidence == 1.0
    client.chat_json.assert_not_called()


def test_no_match_sets_nomap():
    client = MagicMock()
    client.embed.return_value = [[0.0] * 16, [0.0] * 16]
    records = map_ngs([_fee(code="Z999")], [_ngs(refs=())],
                      client=client, embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code is None
    assert records[0].NGS_mapping_method is None


def test_llm_verifies_semantic_candidate():
    client = MagicMock()
    # code not in any NGS code_refs, but semantically matches
    client.embed.return_value = [
        [1.0] + [0.0] * 15,  # fee description embedding
        [1.0] + [0.0] * 15,  # ngs description embedding — cosine = 1.0
    ]
    client.chat_json.return_value = NGSVerdict(accept=True, confidence=0.9)
    records = map_ngs([_fee(code="K999")], [_ngs(refs=())],
                      client=client, embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code == "1AA"
    assert records[0].NGS_mapping_method == "llm"
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_ngs_mapper.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/ngs_mapper.py`:

```python
"""Resolve FSC codes to NGS categories.
Three tiers: exact code_refs match, semantic top-1 with LLM verification, NOMAP."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from src.pipeline.schema import FeeCodeRecord, NGSRecord

SIMILARITY_THRESHOLD = 0.5


class NGSVerdict(BaseModel):
    accept: bool
    confidence: float = Field(ge=0.0, le=1.0)


class _ClientLike(Protocol):
    def embed(self, texts: list[str], *, model: str, dim: int,
              batch_size: int = 500) -> list[list[float]]: ...
    def chat_json(self, *, prompt: str, schema: type, model: str,
                  system: str | None = None, temperature: float = 0.0): ...


def _normalize(vecs: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def map_ngs(
    records: list[FeeCodeRecord],
    ngs: list[NGSRecord],
    *,
    client: _ClientLike,
    embed_model: str,
    llm_model: str,
    dim: int,
) -> list[FeeCodeRecord]:
    # 1. Exact-match pass
    exact_index: dict[str, NGSRecord] = {}
    for n in ngs:
        for ref in n.code_refs:
            exact_index.setdefault(ref.upper(), n)

    out: list[FeeCodeRecord] = []
    unresolved: list[FeeCodeRecord] = []
    for r in records:
        hit = exact_index.get(r.fsc_code.upper())
        if hit is not None:
            out.append(r.model_copy(update={
                "NGS_code": hit.ngs_code,
                "NGS_label": hit.ngs_label,
                "NGS_mapping_method": "exact",
                "NGS_mapping_confidence": 1.0,
            }))
        else:
            unresolved.append(r)

    if not unresolved or not ngs:
        return out + unresolved

    # 2. Semantic pass
    fee_texts = [f"{r.fsc_fn}. {r.fsc_description}" for r in unresolved]
    ngs_texts = [f"{n.ngs_label}. {n.ngs_description}" for n in ngs]
    vecs = client.embed(fee_texts + ngs_texts, model=embed_model, dim=dim)
    arr = _normalize(vecs)
    fee_vecs = arr[: len(fee_texts)]
    ngs_vecs = arr[len(fee_texts):]
    sims = fee_vecs @ ngs_vecs.T  # (F, N)

    for i, r in enumerate(unresolved):
        top_j = int(np.argmax(sims[i]))
        top_sim = float(sims[i, top_j])
        if top_sim < SIMILARITY_THRESHOLD:
            out.append(r)
            continue
        candidate = ngs[top_j]
        verdict = client.chat_json(
            prompt=(
                f"Fee code {r.fsc_code} ({r.province}): {r.fsc_fn}. {r.fsc_description}\n"
                f"Candidate NGS category {candidate.ngs_code}: "
                f"{candidate.ngs_label}. {candidate.ngs_description}\n"
                "Is this the correct category?"
            ),
            schema=NGSVerdict, model=llm_model, temperature=0.0,
        )
        if verdict.accept:
            out.append(r.model_copy(update={
                "NGS_code": candidate.ngs_code,
                "NGS_label": candidate.ngs_label,
                "NGS_mapping_method": "llm",
                "NGS_mapping_confidence": verdict.confidence,
            }))
        else:
            out.append(r)
    return out
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_ngs_mapper.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ngs_mapper.py tests/unit/test_ngs_mapper.py
git commit -m "feat(pipeline): add three-tier NGS mapper"
```

---

## Task 13: OpenAI embedder

**Files:**
- Create: `src/pipeline/embed.py`
- Test: `tests/unit/test_embed.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_embed.py
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.pipeline.schema import FeeCodeRecord
from src.pipeline.embed import build_embeddings, save_npz, load_npz


def _rec(code: str) -> FeeCodeRecord:
    return FeeCodeRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="desc",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
    )


def test_build_embeddings_returns_l2_normalized():
    recs = [_rec("K040"), _rec("K041")]
    client = MagicMock()
    client.embed.return_value = [[3.0, 4.0], [1.0, 0.0]]
    arr, ids = build_embeddings(recs, client=client, model="m", dim=2)
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(arr, axis=1), [1.0, 1.0])
    np.testing.assert_array_equal(ids, [0, 1])


def test_npz_roundtrip(tmp_path: Path):
    arr = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([7, 42], dtype=np.int32)
    out = tmp_path / "e.npz"
    save_npz(out, arr, ids)
    arr2, ids2 = load_npz(out)
    np.testing.assert_array_equal(arr, arr2)
    np.testing.assert_array_equal(ids, ids2)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_embed.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/embed.py`:

```python
"""OpenAI embeddings for canonical records. L2-normalized at write time."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from src.pipeline.schema import FeeCodeRecord


class _ClientLike(Protocol):
    def embed(self, texts: list[str], *, model: str, dim: int,
              batch_size: int = 500) -> list[list[float]]: ...


def _record_text(r: FeeCodeRecord) -> str:
    parts = [
        r.fsc_fn, r.fsc_description, r.fsc_section or "",
        r.fsc_subsection or "", r.fsc_chapter or "", r.NGS_label or "",
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


def build_embeddings(
    records: list[FeeCodeRecord], *, client: _ClientLike, model: str, dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    texts = [_record_text(r) for r in records]
    vecs = client.embed(texts, model=model, dim=dim)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    ids = np.arange(len(records), dtype=np.int32)
    return arr, ids


def save_npz(path: Path, embeddings: np.ndarray, record_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), embeddings=embeddings, record_ids=record_ids)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(str(path), allow_pickle=False)
    return npz["embeddings"].astype(np.float32), npz["record_ids"].astype(np.int32)
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_embed.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/embed.py tests/unit/test_embed.py
git commit -m "feat(pipeline): add OpenAI embedding builder + npz IO"
```

---

## Task 14: Regression gate

**Files:**
- Create: `src/pipeline/regression.py`
- Test: `tests/unit/test_regression.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_regression.py
from src.pipeline.schema import FeeCodeRecord
from src.pipeline.regression import diff, check, DiffReport


def _rec(code: str, province="ON", desc="desc"):
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description=desc,
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
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
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_regression.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/regression.py`:

```python
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
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_regression.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/regression.py tests/unit/test_regression.py
git commit -m "feat(pipeline): add regression diff + fail-loud gate"
```

---

## Task 15: Pipeline orchestrator

**Files:**
- Create: `src/pipeline/run.py`
- Test: `tests/integration/test_run_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_run_skeleton.py
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline.run import run_pipeline, PipelineConfig


@pytest.mark.integration
def test_pipeline_skips_completed_steps(tmp_path: Path, monkeypatch):
    # Pre-create the expected final artifact and the pipeline should skip
    version_dir = tmp_path / "parsed" / "v2026-04-13"
    version_dir.mkdir(parents=True)
    (version_dir / "codes.json").write_text("[]", encoding="utf-8")
    (version_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1", "generated_at": "t", "git_sha": "x",
                    "row_counts": {"ON": 0, "BC": 0, "YT": 0},
                    "source_pdf_hashes": {}, "models": {}}),
        encoding="utf-8",
    )
    cfg = PipelineConfig(
        raw_pdf_dir=tmp_path / "raw" / "pdf",
        raw_docx_dir=tmp_path / "raw" / "docx",
        output_dir=tmp_path / "parsed",
        diagnostics_dir=tmp_path / "diagnostics",
        version="2026-04-13",
        force=False,
    )
    # Should no-op; force=False and artifacts exist
    result = run_pipeline(cfg, client=None)
    assert result.skipped is True
```

- [ ] **Step 2: Run, verify it fails**

```bash
pytest tests/integration/test_run_skeleton.py -v
```

- [ ] **Step 3: Implement**

`src/pipeline/run.py`:

```python
"""Pipeline orchestrator. Idempotent (skip-if-output-exists), --force to redo."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
    golden_set: frozenset[tuple[str, str]] = frozenset()


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


def _province_pdf(dir: Path, province: str) -> Path | None:
    patterns = {
        "ON": "moh-schedule-benefit-*.pdf",
        "BC": "msc_payment_schedule_*.pdf",
        "YT": "yukon_physician_fee_guide_*.pdf",
    }
    hits = sorted(dir.glob(patterns[province]))
    return hits[-1] if hits else None


def run_pipeline(cfg: PipelineConfig, *, client: Optional[OpenAIClient]) -> PipelineResult:
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
            rows, client=client, model=cfg.extract_model,
            context_lines=context_lines, threshold=0.8,
        )
        _write_jsonl(cfg.diagnostics_dir / "unresolved.jsonl",
                     [r.model_dump(mode="json") for r in unresolved], append=True)
        all_candidates.extend(rescued)

    # Validate
    records, rejects = validate(all_candidates)
    _write_jsonl(cfg.diagnostics_dir / "validation_rejects.jsonl",
                 [{"row": e.candidate.model_dump(mode="json"), "reason": e.reason}
                  for e in rejects])

    # NGS mapping
    ngs_records = []
    for docx_path in sorted(cfg.raw_docx_dir.glob("*.docx")):
        ngs_records.extend(parse_ngs_docx(docx_path))
    records = map_ngs(
        records, ngs_records, client=client,
        embed_model=cfg.embed_model, llm_model=cfg.extract_model, dim=cfg.embed_dim,
    )
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

    row_counts = {p: sum(1 for r in records if r.province == p)
                  for p in ("ON", "BC", "YT")}
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
        p for p in output_dir.glob("v*/codes.json")
        if p.parent.name != f"v{current_version}"
    )
    return candidates[-1] if candidates else None
```

- [ ] **Step 4: Verify test passes**

```bash
pytest tests/integration/test_run_skeleton.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/run.py tests/integration/test_run_skeleton.py
git commit -m "feat(pipeline): add idempotent pipeline orchestrator"
```

---

## Task 16: Core loader + core matching (port from app/lookup_engine.py)

**Files:**
- Create: `src/core/__init__.py`, `src/core/loader.py`, `src/core/matching.py`
- Test: `tests/unit/test_core_loader.py`, `tests/unit/test_core_matching.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_core_loader.py`:

```python
import json
from pathlib import Path
import numpy as np

from src.core.loader import load_latest
from src.pipeline.schema import FeeCodeRecord, Manifest


def _rec(province, code):
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description="d",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
    )


def test_load_latest_picks_max_version(tmp_path: Path):
    for version in ["2026-01-01", "2026-04-13"]:
        vdir = tmp_path / f"v{version}"
        vdir.mkdir()
        recs = [_rec("ON", "K040")]
        (vdir / "codes.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in recs])
        )
        np.savez(str(vdir / "embeddings.npz"),
                 embeddings=np.zeros((1, 2), dtype=np.float32),
                 record_ids=np.array([0], dtype=np.int32))
        (vdir / "manifest.json").write_text(Manifest(
            generated_at="t", git_sha="x",
            row_counts={"ON": 1, "BC": 0, "YT": 0},
            source_pdf_hashes={}, models={},
        ).model_dump_json())

    records, emb, ids, manifest = load_latest(tmp_path)
    assert manifest.row_counts["ON"] == 1
```

`tests/unit/test_core_matching.py`:

```python
import numpy as np
import pytest

from src.core.matching import search
from src.pipeline.schema import FeeCodeRecord


def _rec(province, code, desc):
    return FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn=code, fsc_description=desc,
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
        NGS_code="1AA", NGS_label="Exam",
    )


def test_search_returns_anchor_and_matches():
    records = [
        _rec("ON", "K040", "periodic health visit adult"),
        _rec("BC", "01712", "periodic health visit adult consultation"),
        _rec("BC", "02000", "fracture repair leg"),
    ]
    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],   # identical to K040
        [0.0, 1.0],   # unrelated
    ], dtype=np.float32)
    record_ids = np.array([0, 1, 2], dtype=np.int32)

    result = search(
        fsc_code="K040", src="ON", dst="BC", top_n=2,
        records=records, embeddings=embeddings, record_ids=record_ids,
    )
    assert result is not None
    assert result.anchor.fsc_code == "K040"
    assert result.matches[0].fee_code.fsc_code == "01712"
    assert result.matches[0].sim_score > result.matches[1].sim_score


def test_search_missing_code_returns_none():
    records = [_rec("ON", "K040", "x")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    record_ids = np.array([0], dtype=np.int32)
    result = search(
        fsc_code="NOPE", src="ON", dst="BC", top_n=5,
        records=records, embeddings=embeddings, record_ids=record_ids,
    )
    assert result is None
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/unit/test_core_loader.py tests/unit/test_core_matching.py -v
```

- [ ] **Step 3: Implement**

`src/core/__init__.py`: empty.

`src/core/loader.py`:

```python
"""Load the newest versioned pipeline artifact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.schema import FeeCodeRecord, Manifest


def load_latest(
    parsed_dir: Path,
) -> tuple[list[FeeCodeRecord], np.ndarray, np.ndarray, Manifest]:
    version_dirs = sorted(parsed_dir.glob("v*"))
    if not version_dirs:
        raise FileNotFoundError(f"No versioned artifacts in {parsed_dir}")
    vdir = version_dirs[-1]

    codes = json.loads((vdir / "codes.json").read_text(encoding="utf-8"))
    records = [FeeCodeRecord(**c) for c in codes]

    embeddings = np.zeros((0, 0), dtype=np.float32)
    record_ids = np.zeros((0,), dtype=np.int32)
    emb_path = vdir / "embeddings.npz"
    if emb_path.exists():
        npz = np.load(str(emb_path), allow_pickle=False)
        embeddings = npz["embeddings"].astype(np.float32)
        record_ids = npz["record_ids"].astype(np.int32)

    manifest = Manifest.model_validate_json((vdir / "manifest.json").read_text())
    return records, embeddings, record_ids, manifest
```

`src/core/matching.py`:

```python
"""Phase 1: port of the existing cosine + Jaccard matching logic.
Same algorithm as app/lookup_engine.py, new data contract. Phase 2 replaces it."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.pipeline.schema import FeeCodeRecord, Province

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "by",
    "at", "from", "as", "is", "are", "be", "this", "that", "per", "each",
}


@dataclass(frozen=True)
class MatchResult:
    fee_code: FeeCodeRecord
    sim_score: float
    ngs_match: bool
    score_method: str  # "semantic" | "jaccard"


@dataclass(frozen=True)
class LookupResult:
    anchor: FeeCodeRecord
    output_province: Province
    matches: list[MatchResult]
    score_method: str


def _tokenize(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {t for t in text.split() if t and t not in STOPWORDS and len(t) > 2}


def _jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    j = inter / len(a | b)
    o = inter / max(len(a), 1)
    return round(0.4 * j + 0.6 * o, 4)


def _record_text(r: FeeCodeRecord) -> str:
    parts = [r.fsc_fn, r.fsc_description, r.fsc_section or "",
             r.fsc_subsection or "", r.fsc_chapter or "", r.NGS_label or ""]
    return " | ".join(p.strip() for p in parts if p.strip())


def search(
    *,
    fsc_code: str,
    src: Province,
    dst: Province,
    top_n: int,
    records: list[FeeCodeRecord],
    embeddings: np.ndarray,
    record_ids: np.ndarray,
) -> LookupResult | None:
    code = fsc_code.strip().upper()
    anchor_idx = next(
        (i for i, r in enumerate(records)
         if r.province == src and r.fsc_code.upper() == code),
        None,
    )
    if anchor_idx is None:
        return None
    anchor = records[anchor_idx]

    candidate_indices = [i for i, r in enumerate(records) if r.province == dst]
    if not candidate_indices:
        return LookupResult(anchor=anchor, output_province=dst, matches=[],
                            score_method="semantic")

    has_embed = embeddings.size > 0 and len(record_ids) == len(records)
    if has_embed:
        matches = _semantic_search(anchor_idx, anchor, candidate_indices,
                                   records, embeddings, record_ids, top_n)
        method = "semantic"
    else:
        matches = _jaccard_search(anchor, candidate_indices, records, top_n)
        method = "jaccard"
    return LookupResult(anchor=anchor, output_province=dst, matches=matches,
                        score_method=method)


def _semantic_search(
    anchor_idx: int, anchor: FeeCodeRecord, cand_indices: list[int],
    records: list[FeeCodeRecord], embeddings: np.ndarray,
    record_ids: np.ndarray, top_n: int,
) -> list[MatchResult]:
    rec_to_embed = {int(record_ids[j]): j for j in range(len(record_ids))}
    q = embeddings[rec_to_embed[anchor_idx]]
    cand_rows = np.array(
        [rec_to_embed[i] for i in cand_indices if i in rec_to_embed],
        dtype=np.int32,
    )
    cand_vecs = embeddings[cand_rows]
    scores = (cand_vecs @ q).astype(float)
    top = np.argsort(-scores)[:top_n]
    return [
        MatchResult(
            fee_code=records[cand_indices[p]],
            sim_score=float(scores[p]),
            ngs_match=(records[cand_indices[p]].NGS_code == anchor.NGS_code
                       and anchor.NGS_code not in (None, "NOMAP", "")),
            score_method="semantic",
        )
        for p in top
    ]


def _jaccard_search(
    anchor: FeeCodeRecord, cand_indices: list[int],
    records: list[FeeCodeRecord], top_n: int,
) -> list[MatchResult]:
    q_tok = _tokenize(_record_text(anchor))
    scored = []
    for i in cand_indices:
        r = records[i]
        sc = _jaccard_overlap(q_tok, _tokenize(_record_text(r)))
        scored.append((sc, r))
    scored.sort(key=lambda x: -x[0])
    return [
        MatchResult(
            fee_code=r, sim_score=sc,
            ngs_match=(r.NGS_code == anchor.NGS_code
                       and anchor.NGS_code not in (None, "NOMAP", "")),
            score_method="jaccard",
        )
        for sc, r in scored[:top_n]
    ]
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/unit/test_core_loader.py tests/unit/test_core_matching.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/core/ tests/unit/test_core_loader.py tests/unit/test_core_matching.py
git commit -m "feat(core): port cosine+Jaccard matching to src/core"
```

---

## Task 17: Typer CLI

**Files:**
- Create: `src/cli.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli.py
from unittest.mock import patch

from typer.testing import CliRunner

from src.cli import app


def test_run_invokes_pipeline():
    runner = CliRunner()
    with patch("src.cli.run_pipeline") as fake_run, \
         patch("src.cli.OpenAIClient") as fake_client:
        fake_run.return_value = None
        fake_client.return_value = "CLIENT"
        result = runner.invoke(app, ["run", "--version", "2026-04-13"])
    assert result.exit_code == 0
    fake_run.assert_called_once()
```

- [ ] **Step 2: Run, verify it fails**

```bash
pytest tests/unit/test_cli.py -v
```

- [ ] **Step 3: Implement**

`src/cli.py`:

```python
"""Typer-based CLI. Thin wrapper around src/pipeline/run.py."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from dotenv import load_dotenv

from src.openai_client import OpenAIClient
from src.pipeline.run import PipelineConfig, run_pipeline

app = typer.Typer(help="FSC-NGS pipeline CLI")

ROOT = Path(__file__).resolve().parent.parent


@app.command()
def run(
    version: str = typer.Option(
        default_factory=lambda: date.today().isoformat(),
        help="Output version tag (default: today)",
    ),
    force: bool = typer.Option(False, "--force", help="Redo steps with existing outputs"),
    accept_regression: str | None = typer.Option(
        None, "--accept-regression", help="Override regression gate with a reason"
    ),
    embed_dim: int = typer.Option(1024, help="Embedding dimension"),
) -> None:
    """Run the full pipeline."""
    load_dotenv()
    cfg = PipelineConfig(
        raw_pdf_dir=ROOT / "data" / "raw" / "pdf",
        raw_docx_dir=ROOT / "data" / "raw" / "docx",
        output_dir=ROOT / "data" / "parsed",
        diagnostics_dir=ROOT / "data" / "diagnostics" / version,
        version=version,
        force=force,
        accept_regression=accept_regression,
        embed_dim=embed_dim,
    )
    client = OpenAIClient()
    run_pipeline(cfg, client=client)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Verify test passes**

```bash
pytest tests/unit/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): add typer-based pipeline CLI"
```

---

## Task 18: Update app to use src/core

**Files:**
- Modify: `app/main.py`, `app/excel_export.py`
- Delete: `app/lookup_engine.py`

- [ ] **Step 1: Rewrite app/excel_export.py to iterate pydantic fields**

Open `app/excel_export.py`. Replace the hardcoded `COLUMNS` list with dynamic field enumeration:

```python
# app/excel_export.py
"""In-memory openpyxl workbook generator. Columns derived from pydantic schema."""
from __future__ import annotations

import io
from typing import Iterable

from openpyxl import Workbook

from src.core.matching import LookupResult
from src.pipeline.schema import FeeCodeRecord

EXPORT_FIELDS = [
    name for name in FeeCodeRecord.model_fields if name != "schema_version"
] + ["sim_score", "ngs_match", "score_method", "is_anchor"]


def _row(record: FeeCodeRecord, *, sim_score: float | None,
         ngs_match: bool | None, score_method: str | None,
         is_anchor: bool) -> list:
    dumped = record.model_dump(mode="json")
    row = [dumped.get(f, "") for f in FeeCodeRecord.model_fields if f != "schema_version"]
    row.extend([
        sim_score if sim_score is not None else "",
        "yes" if ngs_match else ("no" if ngs_match is False else ""),
        score_method or "",
        "yes" if is_anchor else "no",
    ])
    return row


def build_workbook(results: Iterable[LookupResult]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "FSC Lookup"
    ws.append(EXPORT_FIELDS)
    for result in results:
        ws.append(_row(result.anchor, sim_score=None, ngs_match=None,
                       score_method=result.score_method, is_anchor=True))
        for m in result.matches:
            ws.append(_row(m.fee_code, sim_score=m.sim_score,
                           ngs_match=m.ngs_match, score_method=m.score_method,
                           is_anchor=False))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
```

- [ ] **Step 2: Rewrite app/main.py to use src/core**

Full new contents of `app/main.py`:

```python
"""FSC Cross-Province Lookup — Streamlit App.
Run: streamlit run app/main.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.core.loader import load_latest
from src.core.matching import LookupResult, search
from app.excel_export import build_workbook

load_dotenv()

st.set_page_config(page_title="FSC Cross-Province Lookup", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

PROVINCES = ["ON", "BC", "YT"]


@st.cache_resource(show_spinner="Loading fee code database...")
def _load():
    records, embeddings, record_ids, manifest = load_latest(
        ROOT / "data" / "parsed"
    )
    return records, embeddings, record_ids, manifest


records, embeddings, record_ids, manifest = _load()
available_provinces = sorted({r.province for r in records})

with st.sidebar:
    st.title("🏥 FSC Lookup")
    st.caption("Cross-province fee code mapping via NGS")
    st.divider()
    src = st.selectbox("Input province", options=available_provinces, index=0)
    dst = st.selectbox(
        "Output province",
        options=[p for p in available_provinces if p != src],
        index=0,
    )
    fsc_input = st.text_input(
        "FSC code", placeholder="e.g. K040, 01712, 0615"
    ).strip().upper()
    top_n = st.slider("Max matches", min_value=1, max_value=10, value=5)
    st.divider()
    st.caption(
        f"Embeddings: **{manifest.models.get('embed', 'unknown')}** "
        f"({embeddings.shape[1] if embeddings.size else '—'}-dim)"
    )
    st.caption(f"Data version: **{manifest.generated_at[:10]}**")

st.title("FSC Cross-Province Lookup")

if not fsc_input:
    st.info("Enter an FSC code in the sidebar to begin.")
    with st.expander("Sample codes to try"):
        cols = st.columns(len(available_provinces))
        for col, prov in zip(cols, available_provinces):
            codes = sorted({r.fsc_code for r in records if r.province == prov})[:15]
            col.markdown(f"**{prov}**")
            col.code("\n".join(codes))
    st.stop()

result: LookupResult | None = search(
    fsc_code=fsc_input, src=src, dst=dst, top_n=top_n,
    records=records, embeddings=embeddings, record_ids=record_ids,
)

if result is None:
    st.error(f"Code **{fsc_input}** not found in **{src}**.")
    st.stop()

anchor = result.anchor
left, _, right = st.columns([10, 1, 10])

with left:
    st.markdown(f"### [{anchor.province}] `{anchor.fsc_code}` "
                f"· NGS {anchor.NGS_code or 'NOMAP'}")
    st.markdown(f"**{anchor.fsc_fn or '—'}**")
    if anchor.fsc_description:
        st.write(anchor.fsc_description)
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${anchor.price}" if anchor.price else "—")
    c2.metric("Page", anchor.page or "—")
    c3.metric("Conf", f"{anchor.extraction_confidence:.2f}")

with right:
    st.markdown(f"### Closest in **{dst}** · {result.score_method}")
    if not result.matches:
        st.info(f"No matches found in {dst}.")
    else:
        for rank, mr in enumerate(result.matches, 1):
            fc = mr.fee_code
            with st.container(border=True):
                st.markdown(
                    f"**#{rank} [{fc.province}] `{fc.fsc_code}`** · "
                    f"NGS {fc.NGS_code or '—'} · "
                    f"{'same NGS' if mr.ngs_match else 'diff NGS'} · "
                    f"{mr.sim_score * 100:.1f}%"
                )
                st.markdown(f"**{fc.fsc_fn or '—'}**")
                if fc.fsc_description:
                    st.caption(fc.fsc_description[:220])

st.divider()
if st.button("Generate Excel", type="primary"):
    xlsx_bytes = build_workbook([result])
    st.download_button(
        label="Download Excel",
        data=xlsx_bytes,
        file_name=f"fsc_{anchor.fsc_code}_{anchor.province}_to_{dst}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
```

- [ ] **Step 3: Delete app/lookup_engine.py**

```bash
git rm app/lookup_engine.py
```

- [ ] **Step 4: Manual smoke test**

If `data/parsed/v*/codes.json` already exists:
```bash
streamlit run app/main.py
```

Open `http://localhost:8501`, confirm the sidebar loads, enter a known code (e.g. `K040`), confirm matches appear. If no versioned data exists yet, skip this step — it will be re-verified in Task 20.

- [ ] **Step 5: Commit**

```bash
git add app/
git commit -m "refactor(app): port Streamlit app to src/core, delete lookup_engine"
```

---

## Task 19: Golden set fixture and integration test

**Files:**
- Create: `tests/fixtures/golden_codes.json`, `tests/integration/test_golden_set.py`

- [ ] **Step 1: Curate the golden set**

Create `tests/fixtures/golden_codes.json` with 60 records (20 per province). Use this stratified template — pick 2-3 codes per chapter from each province's fee schedule. Example entries (the engineer fills in the real 60 during task execution using `data/parsed/fsc_ngs_mapped.json` as reference):

```json
[
  {"province": "ON", "fsc_code": "K040", "expected_chapter_contains": "general",
   "expected_description_contains": ["periodic", "visit"], "expected_price_min": 50.0},
  {"province": "ON", "fsc_code": "A007", "expected_description_contains": ["consultation"]},
  {"province": "BC", "fsc_code": "01712", "expected_description_contains": ["consultation"]},
  {"province": "YT", "fsc_code": "0615", "expected_description_contains": ["visit"]}
]
```

Rule for curation: from the current `data/parsed/fsc_ngs_mapped.json` take 20 codes per province distributed across chapters (stratified by `fsc_chapter` if available, else uniform random with `random.seed(42)`). Record their core description substring and price floor.

- [ ] **Step 2: Write the integration test**

```python
# tests/integration/test_golden_set.py
import json
from pathlib import Path

import pytest

from src.core.loader import load_latest

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.integration
def test_golden_set_codes_all_present():
    try:
        records, _, _, _ = load_latest(ROOT / "data" / "parsed")
    except FileNotFoundError:
        pytest.skip("No pipeline artifacts yet")

    golden = json.loads((ROOT / "tests" / "fixtures" / "golden_codes.json").read_text())
    index = {(r.province, r.fsc_code): r for r in records}

    missing = [
        (g["province"], g["fsc_code"])
        for g in golden
        if (g["province"], g["fsc_code"]) not in index
    ]
    assert not missing, f"Missing golden-set codes: {missing}"

    for g in golden:
        r = index[(g["province"], g["fsc_code"])]
        desc = r.fsc_description.lower()
        for needle in g.get("expected_description_contains", []):
            assert needle.lower() in desc, (
                f"{g['province']}:{g['fsc_code']} description '{desc[:60]}' "
                f"missing '{needle}'"
            )
        if "expected_price_min" in g and r.price is not None:
            assert float(r.price) >= g["expected_price_min"]
```

- [ ] **Step 3: Run full pipeline end-to-end**

```bash
python -m src.cli run --version 2026-04-13
```

Expected: pipeline completes in <10 minutes, writes `data/parsed/v2026-04-13/{codes.json, embeddings.npz, manifest.json}`, diagnostics in `data/diagnostics/2026-04-13/`. Total cost <$1 per `data/diagnostics/2026-04-13/costs.json`.

If the pipeline fails: inspect `data/diagnostics/2026-04-13/validation_rejects.jsonl` and `unresolved.jsonl`. If the reject rate exceeds 1% for any province, the structural extractor for that province needs tuning — revisit Task 6/7/8.

- [ ] **Step 4: Run golden set test**

```bash
pytest tests/integration/test_golden_set.py -v
```

Expected: all 60 codes present, descriptions pass. If any codes are missing, add them to `data/diagnostics/2026-04-13/unresolved.jsonl` analysis and fix the structural rules.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/golden_codes.json tests/integration/test_golden_set.py data/parsed/v2026-04-13/codes.json data/parsed/v2026-04-13/manifest.json
git commit -m "test(pipeline): add 60-code golden set fixture + integration test"
```

Note: `embeddings.npz` is gitignored per Task 1. `codes.json` and `manifest.json` are committed.

---

## Task 20: Property + snapshot tests

**Files:**
- Create: `tests/property/test_schema_properties.py`, `tests/regression/test_snapshot.py`

- [ ] **Step 1: Write the property test**

```python
# tests/property/test_schema_properties.py
from decimal import Decimal

import pytest
from hypothesis import given, strategies as st

from src.pipeline.schema import FeeCodeRecord


VALID_PROVINCE = st.sampled_from(["ON", "BC", "YT"])
VALID_CODE = st.text(min_size=3, max_size=5, alphabet="ABCDEFGHIJK0123456789")
CONFIDENCE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@pytest.mark.property
@given(
    province=VALID_PROVINCE, code=VALID_CODE, page=st.integers(min_value=1, max_value=2000),
    confidence=CONFIDENCE,
    price=st.one_of(st.none(), st.decimals(min_value=Decimal("0.01"),
                                           max_value=Decimal("9999.99"),
                                           places=2, allow_nan=False, allow_infinity=False)),
)
def test_record_json_roundtrip(province, code, page, confidence, price):
    r = FeeCodeRecord(
        province=province, fsc_code=code, fsc_fn="fn", fsc_description="d",
        page=page, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=confidence, price=price,
    )
    data = r.model_dump(mode="json")
    r2 = FeeCodeRecord(**data)
    assert r == r2
```

- [ ] **Step 2: Write the snapshot test**

```python
# tests/regression/test_snapshot.py
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT = ROOT / "tests" / "regression" / "snapshot_codes.json"


@pytest.mark.regression
def test_codes_match_snapshot():
    latest_dir = sorted((ROOT / "data" / "parsed").glob("v*"))[-1]
    current = json.loads((latest_dir / "codes.json").read_text(encoding="utf-8"))

    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(current, indent=2), encoding="utf-8")
        pytest.skip("Snapshot created; re-run to compare")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    missing = [(r["province"], r["fsc_code"]) for r in expected
               if (r["province"], r["fsc_code"]) not in
               {(c["province"], c["fsc_code"]) for c in current}]
    added = [(r["province"], r["fsc_code"]) for r in current
             if (r["province"], r["fsc_code"]) not in
             {(c["province"], c["fsc_code"]) for c in expected}]
    assert not missing, f"codes removed vs snapshot: {missing[:10]} (total {len(missing)})"
    assert len(added) < 0.05 * len(expected), (
        f"too many new codes vs snapshot: {len(added)}"
    )
```

- [ ] **Step 3: Run property tests**

```bash
pytest tests/property/ -v
```

Expected: all pass (hypothesis generates ~100 examples per test).

- [ ] **Step 4: Run snapshot test** (twice — first seeds, second compares)

```bash
pytest tests/regression/ -v    # first run: creates snapshot, reports skip
pytest tests/regression/ -v    # second run: compares, should pass
```

- [ ] **Step 5: Commit**

```bash
git add tests/property/ tests/regression/test_snapshot.py tests/regression/snapshot_codes.json
git commit -m "test(pipeline): add property + snapshot regression tests"
```

---

## Task 21: Cleanup and docs

**Files:**
- Delete: old pipeline modules (listed in File Map § Deleted)
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Delete obsolete source files**

```bash
git rm src/extract_mistral.py src/parse_mistral.py src/extract_pdfs.py src/parse_all_provinces.py src/parse_docx_full.py src/map_fsc_ngs.py src/cross_province.py src/build_embeddings.py
git rm scripts/run_pipeline.py scripts/reextract_zeros.py
rmdir scripts 2>/dev/null || true
```

- [ ] **Step 2: Update README.md quick-start section**

Replace the "Quick start" and "Project structure" sections of `README.md` with:

```markdown
## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API key

Copy `.env.example` to `.env` and set your key:
```bash
cp .env.example .env
# edit .env, set OPENAI_API_KEY
```

### 3. Run the data pipeline

Place the raw PDFs in `data/raw/pdf/` and reference DOCXs in `data/raw/docx/`, then:

```bash
python -m src.cli run --version 2026-04-13
```

This runs the structural extractor, LLM rescue for ambiguous rows, NGS mapping, and embedding build in one pass. Typical runtime is <10 minutes with no GPU; typical cost is <$1 in OpenAI API fees.

### 4. Launch the app

```bash
streamlit run app/main.py
```

## Project structure

```
fsc-ngs-ai/
├── app/
│   ├── main.py            # Streamlit UI
│   └── excel_export.py    # Workbook generation
├── src/
│   ├── pipeline/          # Offline data pipeline (writes data/parsed/)
│   ├── core/              # Runtime matching (reads data/parsed/)
│   ├── openai_client.py   # Single OpenAI chokepoint
│   └── cli.py             # Typer CLI
├── data/
│   ├── raw/               # Source PDFs + DOCXs (not committed)
│   └── parsed/v<DATE>/    # Versioned artifacts; codes.json + manifest committed
├── tests/                 # unit / integration / property / regression
└── requirements.txt
```
```

- [ ] **Step 3: Update CLAUDE.md**

Replace the "Common commands" and "Architecture" sections to reflect the new pipeline. Key edits:
- Replace `python scripts/run_pipeline.py` with `python -m src.cli run`
- Replace the Mistral OCR data-flow diagram with the new pymupdf→structural→(semantic rescue)→validate→NGS→embed flow
- Remove references to `src/extract_pdfs.py`, `src/parse_all_provinces.py`, `src/build_embeddings.py`, `scripts/reextract_zeros.py`, `src/extract_mistral.py`, `src/parse_mistral.py`
- Update the runtime section: `app/lookup_engine.py` is gone; runtime lives in `src/core/matching.py` and `src/core/loader.py`
- Remove the BGE/CUDA section; add a short OpenAI section: embeddings come from `text-embedding-3-large` at 1024-dim, configured via `OPENAI_EMBED_MODEL` and `OPENAI_EMBED_DIM` in `.env`
- Update the "Record schema" section: pydantic `FeeCodeRecord` in `src/pipeline/schema.py` is the source of truth; `excel_export.EXPORT_FIELDS` derives from it
- Remove the "`db/mapping.db` is a stale artifact" note since that file is unrelated to this rebuild
- Add a reference to the design doc: "Phase 1 spec: `docs/superpowers/specs/2026-04-13-fsc-ngs-rebuild-design.md`"

- [ ] **Step 4: Full test suite**

```bash
pytest -q
```

Expected: all unit, integration, property, and regression tests pass.

- [ ] **Step 5: Commit and open PR**

```bash
git add -A
git commit -m "chore: remove obsolete Mistral/pymupdf-coord modules, update docs"
git push -u origin rebuild/pipeline-openai
gh pr create --title "Phase 1: Rebuild pipeline (OpenAI, no GPU, no Mistral)" \
    --body-file docs/superpowers/specs/2026-04-13-fsc-ngs-rebuild-design.md
```

---

## Self-Review Notes

**Spec coverage:**
- §2 Contract A (canonical `FeeCodeRecord`) → Task 2
- §2 Contract B (matching signature) → Task 16
- §2 Contract C (.env config) → Tasks 1, 17
- §2 Contract D (versioned artifacts) → Tasks 15, 16
- §4.1 Success criteria 1-6 → Tasks 19, 20 (golden set, snapshot, cost in diagnostics, no torch in req.txt)
- §4.2 new source layout → Tasks 2-17 collectively
- §4.3.1 schema → Task 2
- §4.3.2 per-province extractors + confidence scoring → Tasks 5, 6, 7, 8
- §4.3.3 semantic rescue (threshold 0.8) → Tasks 5 (Confidence), 9
- §4.3.4 validator → Task 10
- §4.3.5 NGS parser → Task 11
- §4.3.6 NGS mapper (three tiers) → Task 12
- §4.3.7 embedder (1024-dim, L2-normalized) → Task 13
- §4.3.8 regression gate → Task 14
- §4.3.9 OpenAI client chokepoint → Task 3
- §4.4 testing pyramid → Tasks 19, 20 (unit tests live per-component)
- §4.5 YAGNI — app port preserves current algorithm → Task 16, 18
- §5 dependency changes → Task 1
- §6 success evidence → Task 19 (end-to-end pipeline run) + Task 21 (no-torch in req.txt)
- §7 open questions — golden set curation rule documented in Task 19 step 1

**Placeholder scan:** No TBDs, no "similar to Task N" references, each task has complete code. Golden-set JSON in Task 19 is an actionable curation rule (stratified sample from the existing mapped data), not a placeholder.

**Type consistency:** `FeeCodeRecord`, `CandidateRow`, `NGSRecord`, `Manifest`, `Province`, `MatchResult`, `LookupResult`, `PipelineConfig`, `DiffReport`, `NGSVerdict`, `RescueOutput`, `CostTracker` — each type is defined once, used consistently across tasks. Method signatures for `search`, `map_ngs`, `rescue`, `validate`, `build_embeddings`, `diff`, `check`, `run_pipeline` match between their definitions and every caller.
