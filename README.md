# FSC Cross-Province Lookup

A web application that maps physician **Fee Service Codes (FSC)** across Canadian provinces and links them to **CIHI National Grouping System (NGS)** categories.

**Use case:** A user provides one FSC code from one province (e.g. `K040` from Ontario). The app finds the closest equivalent codes in the other provinces and shows the NGS category for all of them — replacing what was previously a manual, hours-long search process.

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

### 2. Run the data pipeline (first time only)

Place the raw PDF and DOCX files in `data/raw/pdf/` and `data/raw/docx/`, then:

```bash
python scripts/run_pipeline.py
```

This runs all steps in order:
1. Extract text layout from PDFs
2. Parse fee codes per province
3. Parse NGS reference DOCX files
4. Map each FSC code to an NGS category
5. Build cross-province groupings

### 3. Launch the app

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

---

## How to use the app

1. Select the **anchor province** in the sidebar
2. Enter an **FSC code** (e.g. `K040`)
3. View the anchor code details and the best-matching codes in the other two provinces
4. Click **Generate Excel** to download a formatted spreadsheet

---

## Project structure

```
fsc-ngs-ai/
├── app/
│   ├── main.py            # Streamlit UI (entry point)
│   ├── lookup_engine.py   # Core search logic (no UI deps)
│   └── excel_export.py    # Excel generation
├── src/
│   ├── extract_pdfs.py        # PDF -> layout JSON
│   ├── parse_all_provinces.py # Layout JSON -> fee codes
│   ├── parse_docx_full.py     # DOCX -> NGS reference data
│   ├── map_fsc_ngs.py         # Fee codes -> NGS mapping
│   └── cross_province.py      # Cross-province grouping
├── scripts/
│   └── run_pipeline.py    # One-command pipeline runner
├── data/
│   ├── raw/               # Source PDFs + DOCXs (not committed)
│   ├── extracted/         # Intermediate layout files (not committed)
│   └── parsed/
│       └── fsc_ngs_mapped.json   # App database (committed)
└── requirements.txt
```

---

## Matching logic

Each code is matched across provinces using:

1. **NGS category** — codes in the same NGS bucket are prioritised
2. **Text similarity** — Jaccard + overlap score on tokenised descriptions
3. **Confidence scoring** — based on presence of function name and price

Similarity score ranges from 0 (no overlap) to 1.0 (identical description).

---

## Future improvements

- [ ] LLM-enhanced rationale generation (OpenAI API or local Ollama/LM Studio)
- [ ] Additional provinces (AB, QC, NS, ...)
- [ ] Batch lookup (upload a CSV of FSC codes)
- [ ] REST API endpoint for programmatic access
- [ ] Docker deployment
