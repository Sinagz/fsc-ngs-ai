# FSC Cross-Province Lookup

A web app that maps physician **Fee Service Codes (FSC)** across Canadian provinces and links them to **CIHI National Grouping System (NGS)** categories.

**Use case:** A user provides one FSC code from one province (e.g. `K040` from Ontario). The app returns the closest equivalent codes in the other provinces and shows the NGS category for all of them — replacing what was previously a manual, hours-long search.

---

## Provinces covered

| Province | Source |
|----------|--------|
| Ontario (ON) | MOH Schedule of Benefits |
| British Columbia (BC) | MSC Payment Schedule |
| Yukon (YT) | Yukon Physician Fee Guide |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set the OpenAI API key

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY=sk-...
```

Optional overrides (defaults shown):
- `OPENAI_EMBED_MODEL=text-embedding-3-large`
- `OPENAI_EMBED_DIM=1024`
- `OPENAI_EXTRACT_MODEL=gpt-4o-mini`

### 3. Run the data pipeline

Place raw PDFs in `data/raw/pdf/` and reference DOCXs in `data/raw/docx/`, then:

```bash
python -m src.cli run
```

This produces `data/parsed/v<DATE>/{codes.json, embeddings.npz, manifest.json}` in one pass: structural extraction (pymupdf) → LLM rescue for ambiguous rows (gpt-4o-mini) → schema validation → NGS mapping (exact + semantic + LLM verdict) → OpenAI embeddings → regression check.

Typical runtime: under 10 minutes on a laptop with no GPU. Typical cost: under $1 in OpenAI fees per full run. Re-runs are cached on disk via hishel, so iterating downstream is free.

Add `--force` to redo even if outputs already exist. Add `--accept-regression "reason"` to override the regression gate (justification stored in the manifest).

### 4. Launch the app

```bash
streamlit run app/main.py
```

Open `http://localhost:8501`.

---

## Project structure

```
fsc-ngs-ai/
├── app/
│   ├── main.py            # Streamlit UI (entry point)
│   └── excel_export.py    # Workbook generator (schema-driven columns)
├── src/
│   ├── pipeline/          # Offline pipeline — only writer of data/parsed/
│   │   ├── schema.py        # Canonical pydantic types (single source of truth)
│   │   ├── io.py            # PDF loader (pymupdf)
│   │   ├── structural/      # Per-province extractors (ontario/bc/yukon)
│   │   ├── semantic.py      # gpt-4o-mini rescue for low-confidence rows
│   │   ├── validate.py      # Strict pydantic gate + uniqueness
│   │   ├── ngs_parser.py    # DOCX → NGSRecord
│   │   ├── ngs_mapper.py    # FSC → NGS (exact / semantic / LLM)
│   │   ├── embed.py         # OpenAI embeddings, L2-normalized
│   │   ├── regression.py    # Diff vs prior version, fail loud
│   │   └── run.py           # Idempotent orchestrator
│   ├── core/              # Runtime — only reader of data/parsed/
│   │   ├── loader.py        # Load newest versioned bundle
│   │   └── matching.py      # search() — cosine + Jaccard fallback
│   ├── openai_client.py   # Single chokepoint for OpenAI calls (httpx + hishel)
│   └── cli.py             # Typer CLI (`python -m src.cli run`)
├── data/
│   ├── raw/               # Source PDFs + DOCXs (gitignored)
│   ├── parsed/v<DATE>/    # Versioned artifacts (codes.json + manifest.json committed; embeddings.npz gitignored)
│   └── diagnostics/<DATE>/ # Per-run rejects, costs, regression diff (gitignored)
├── tests/                 # unit / integration / property / regression
├── docs/superpowers/      # Design spec + implementation plan
└── requirements.txt
```

---

## Matching logic (Phase 1)

1. **Semantic similarity** — cosine over OpenAI embeddings (`text-embedding-3-large`, default 1024-dim).
2. **Jaccard fallback** — when embeddings are missing, blended Jaccard + overlap on tokenized descriptions.
3. **NGS flag** — same-NGS is surfaced as an informational badge but does NOT boost ranking. Two provinces can legitimately assign equivalent procedures to different NGS codes.

---

## Tests

```bash
pytest                  # all (unit + integration + property + regression)
pytest -m unit          # fast, no I/O
pytest -m integration   # full pipeline against fixtures or real artifacts
pytest -m regression    # snapshot diff against tests/regression/snapshot_codes.json
```

Integration and regression tests skip cleanly when there are no `data/parsed/v*/` artifacts.

---

## Future work (sketched)

- **Phase 2:** OpenAI-embedding retrieval + `gpt-4o` reranking with structured rationale.
- **Phase 3:** Streamed rationale generation, natural-language search, "why not X?" drill-downs.
- **Phase 4:** FastAPI service + DuckDB/SQLite storage, Docker.
- Additional provinces (AB, QC, NS, ...).
- Batch lookup (CSV upload).
