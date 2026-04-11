"""
Mistral OCR-based fee code extractor.

Two-pass pipeline per province:
  Pass 1 — OCR:       Upload PDF once -> sliding 8-page windows -> per-page markdown
  Pass 2 — Extract:   Sliding window over markdown -> mistral-small structured JSON
  Pass 3 — Merge:     Deduplicate (first occurrence wins = definition page beats index)

Output: data/extracted/mistral/{prov}_codes.json
  [{fsc_code, fsc_fn, fsc_description, price, chapter, section, page_start}]

Usage:
    python src/extract_mistral.py [--province ON|BC|YT] [--skip-ocr] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from mistralai.client import Mistral

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR  = BASE_DIR / "data" / "raw" / "pdf"
OUT_DIR  = BASE_DIR / "data" / "extracted" / "mistral"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVINCE_PDFS = {
    "ON": "moh-schedule-benefit-2025-03-19 (Nov 18, 2025).pdf",
    "BC": "msc_payment_schedule_-_dec_31_2024 (May 29, 2025).pdf",
    "YT": "yukon_physician_fee_guide_2024 (Nov 18, 2025).pdf",
}

WINDOW   = 8    # pages per OCR call
OVERLAP  = 2    # pages shared between consecutive windows
STEP     = WINDOW - OVERLAP  # 6 pages advance per step

OCR_MODEL    = "mistral-ocr-latest"
CHAT_MODEL   = "mistral-small-latest"
RETRY_WAIT      = 10   # seconds between retries
MAX_RETRIES     = 4
SIGNED_URL_TTL  = 20   # refresh signed URL every N windows (expires after ~1h)


# ── Fee code extraction schema ────────────────────────────────────────────────

FEE_CODE_SCHEMA = {
    "type": "object",
    "title": "FeeCodeWindow",
    "properties": {
        "chapter": {
            "type": "string",
            "description": "Current chapter heading visible in this page range (e.g. 'General Surgery', 'Consultations'). Empty string if none visible."
        },
        "section": {
            "type": "string",
            "description": "Current section heading (sub-heading under chapter). Empty string if none."
        },
        "codes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fsc_code": {
                        "type": "string",
                        "description": "The fee/service code exactly as printed (e.g. K040, S207, V72657, 00158)"
                    },
                    "fsc_fn": {
                        "type": "string",
                        "description": "Short name or function of this fee code (the label/heading for this code)"
                    },
                    "fsc_description": {
                        "type": "string",
                        "description": "Any additional descriptive text, conditions, or notes for this code. Empty string if none."
                    },
                    "price": {
                        "type": "string",
                        "description": "Fee amount as printed (e.g. '125.50', '$125.50'). Empty string if not present."
                    }
                },
                "required": ["fsc_code", "fsc_fn"]
            }
        }
    },
    "required": ["codes"]
}

EXTRACTION_SYSTEM = (
    "You are a medical billing assistant. Extract physician fee code entries "
    "from Canadian fee schedule pages. Always return valid JSON."
)

EXTRACTION_PROMPT = """\
Extract all physician fee codes visible in the text below.

Return JSON exactly like this:
{{
  "chapter": "<current chapter heading, or empty string>",
  "section": "<current section heading, or empty string>",
  "codes": [
    {{"fsc_code": "B 0010", "fsc_fn": "Intramuscular medications", "fsc_description": "", "price": "17.90"}},
    ...
  ]
}}

Rules:
- fsc_code: the billing code exactly as printed (e.g. K040, S207, B 0010, 00158, A001)
- fsc_fn: the short name or label for the code
- fsc_description: any additional conditions or notes (empty string if none)
- price: dollar amount as a string, no $ sign (empty string if not shown)
- Include ALL codes — even if the page is a table, list, or mixed format
- If no fee codes are present (e.g. table of contents, preamble only), return {{"chapter":"","section":"","codes":[]}}

Fee schedule text (pages {start_page}-{end_page}):

{markdown}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e)[:120]
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {err_str}")
            time.sleep(RETRY_WAIT * (attempt + 1))


def _get_signed_url(client: Mistral, file_id: str) -> str:
    signed = _call_with_retry(client.files.get_signed_url, file_id=file_id)
    return signed.url


def _ocr_window(client: Mistral, signed_url: str, pages: list[int]) -> list[dict]:
    """Call Mistral OCR for a specific page range. Returns list of {index, markdown}."""
    resp = _call_with_retry(
        client.ocr.process,
        model=OCR_MODEL,
        document={"type": "document_url", "document_url": signed_url},
        pages=pages,
    )
    return [{"index": p.index, "markdown": p.markdown} for p in resp.pages]


MAX_MARKDOWN_CHARS = 24_000   # ~3000 chars/page * 8 pages


def _extract_codes(client: Mistral, markdown: str, start_page: int, end_page: int) -> dict:
    """Call mistral-small to extract fee codes from OCR markdown."""
    # Truncate if too long (preamble pages can be very dense)
    if len(markdown) > MAX_MARKDOWN_CHARS:
        markdown = markdown[:MAX_MARKDOWN_CHARS] + "\n...[truncated]"

    prompt = EXTRACTION_PROMPT.format(
        start_page=start_page,
        end_page=end_page,
        markdown=markdown,
    )
    resp = _call_with_retry(
        client.chat.complete,
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"    [warn] JSON decode failed for pages {start_page}-{end_page}")
        return {"codes": [], "chapter": "", "section": ""}


# ── main per-province pipeline ────────────────────────────────────────────────

def extract_province(province: str, client: Mistral, skip_ocr: bool, force: bool):
    pdf_path = PDF_DIR / PROVINCE_PDFS[province]
    if not pdf_path.exists():
        print(f"[{province}] PDF not found: {pdf_path.name} — skipping")
        return

    ocr_path   = OUT_DIR / f"{province.lower()}_ocr.json"
    codes_path = OUT_DIR / f"{province.lower()}_codes.json"

    # ── Pass 1: OCR ───────────────────────────────────────────────────────────
    if skip_ocr and ocr_path.exists():
        print(f"[{province}] Loading cached OCR from {ocr_path.name}")
        all_pages = json.loads(ocr_path.read_text(encoding="utf-8"))
    else:
        if codes_path.exists() and not force:
            print(f"[{province}] Already done ({codes_path.name}) — use --force to re-run")
            return

        print(f"[{province}] Uploading PDF ({pdf_path.stat().st_size // 1024 // 1024} MB)...")
        with open(pdf_path, "rb") as f:
            uploaded = _call_with_retry(
                client.files.upload,
                file={"file_name": pdf_path.name, "content": f},
                purpose="ocr",
            )
        file_id = uploaded.id
        print(f"[{province}] Uploaded -> file_id={file_id}")
        time.sleep(1)
        signed_url = _get_signed_url(client, file_id)
        print(f"[{province}] Signed URL obtained")

        # Count pages
        import fitz
        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        doc.close()
        print(f"[{province}] {n_pages} pages, window={WINDOW}, step={STEP}")

        # Load checkpoint if a previous run was interrupted
        ckpt_path = ocr_path.with_suffix(".ckpt.json")
        all_pages: dict[int, str] = {}  # page_index -> markdown
        if ckpt_path.exists():
            ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            all_pages = {p["index"]: p["markdown"] for p in ckpt_data}
            print(f"[{province}] Resumed from checkpoint: {len(all_pages)} pages already done")
        windows = list(range(0, n_pages, STEP))
        for wi, start in enumerate(windows):
            # Refresh signed URL periodically (expires after ~1h)
            if wi % SIGNED_URL_TTL == 0 and wi > 0:
                signed_url = _get_signed_url(client, file_id)

            pages = list(range(start, min(start + WINDOW, n_pages)))
            # Skip if all pages already collected (overlap region)
            new_pages = [p for p in pages if p not in all_pages]
            if not new_pages:
                continue

            print(f"  [{province}] OCR window {wi+1}/{len(windows)}: pages {pages[0]}-{pages[-1]}", end=" ", flush=True)
            t0 = time.time()
            page_results = _ocr_window(client, signed_url, pages)
            for pr in page_results:
                all_pages[pr["index"]] = pr["markdown"]
            print(f"({time.time()-t0:.1f}s, {len(page_results)} pages)")

            # Save checkpoint every 10 windows so a crash doesn't lose everything
            if (wi + 1) % 10 == 0:
                ckpt = [{"index": i, "markdown": all_pages[i]} for i in sorted(all_pages)]
                ocr_path.with_suffix(".ckpt.json").write_text(
                    json.dumps(ckpt, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            time.sleep(0.3)  # gentle rate limit

        # Save OCR cache (list sorted by page index)
        pages_list = [{"index": i, "markdown": all_pages[i]}
                      for i in sorted(all_pages)]
        ocr_path.write_text(json.dumps(pages_list, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{province}] OCR saved -> {ocr_path.name} ({len(pages_list)} pages)")

        # Clean up uploaded file
        try:
            client.files.delete(file_id=file_id)
            print(f"[{province}] Deleted remote file {file_id}")
        except Exception:
            pass

    # ── Pass 2: Extract structured codes from OCR markdown ───────────────────
    if codes_path.exists() and not force:
        print(f"[{province}] Codes already extracted ({codes_path.name}) — use --force to re-run")
        return

    # Rebuild page_markdown dict
    if isinstance(all_pages, list):
        page_md: dict[int, str] = {p["index"]: p["markdown"] for p in all_pages}
    else:
        page_md = all_pages

    n_pages = max(page_md) + 1
    windows = list(range(0, n_pages, STEP))
    all_raw: list[dict] = []

    # Load extraction checkpoint if present
    extract_ckpt_path = OUT_DIR / f"{province.lower()}_extract.ckpt.json"
    done_windows: set[int] = set()
    if extract_ckpt_path.exists():
        ckpt_data = json.loads(extract_ckpt_path.read_text(encoding="utf-8"))
        all_raw = ckpt_data.get("raw", [])
        done_windows = set(ckpt_data.get("done_windows", []))
        print(f"[{province}] Resumed extraction: {len(done_windows)} windows done, {len(all_raw)} codes so far")

    print(f"[{province}] Extracting fee codes from {len(windows)} windows...")
    for wi, start in enumerate(windows):
        if start in done_windows:
            continue  # already done in a previous run

        end = min(start + WINDOW - 1, n_pages - 1)
        pages = list(range(start, end + 1))
        combined_md = "\n\n---PAGE BREAK---\n\n".join(
            page_md.get(p, "") for p in pages
        )
        if not combined_md.strip():
            done_windows.add(start)
            continue

        print(f"  [{province}] Extract window {wi+1}/{len(windows)}: pages {start}-{end}", end=" ", flush=True)
        t0 = time.time()
        try:
            result = _extract_codes(client, combined_md, start, end)
        except Exception as e:
            print(f"\n    [SKIP] window {start}-{end} failed after retries: {str(e)[:80]}")
            done_windows.add(start)
            continue
        codes   = result.get("codes", [])
        chapter = result.get("chapter", "")
        section = result.get("section", "")

        for c in codes:
            c["chapter"]    = c.get("chapter") or chapter
            c["section"]    = c.get("section") or section
            c["page_start"] = start
            c["province"]   = province

        all_raw.extend(codes)
        done_windows.add(start)
        print(f"({time.time()-t0:.1f}s, {len(codes)} codes)")

        # Save extraction checkpoint every 5 windows
        if (wi + 1) % 5 == 0:
            extract_ckpt_path.write_text(
                json.dumps({"raw": all_raw, "done_windows": list(done_windows)},
                           ensure_ascii=False), encoding="utf-8"
            )
        time.sleep(0.2)

    # ── Pass 3: Deduplicate — first occurrence wins ───────────────────────────
    seen: dict[str, dict] = {}
    for entry in all_raw:
        code = (entry.get("fsc_code") or "").strip().upper()
        if not code:
            continue
        if code not in seen:
            entry["fsc_code"] = code
            seen[code] = entry

    deduped = list(seen.values())
    codes_path.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{province}] Done — {len(all_raw)} raw -> {len(deduped)} unique codes -> {codes_path.name}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract fee codes via Mistral OCR")
    parser.add_argument("--province", choices=["ON", "BC", "YT"],
                        help="Run for a single province (default: all)")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="Skip OCR pass if cached JSON exists")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output files already exist")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY not set — add it to .env or environment")

    # Use a long timeout — OCR on dense pages can take 2-3 min
    http_client = httpx.Client(timeout=httpx.Timeout(300.0))
    client = Mistral(api_key=api_key, client=http_client)

    provinces = [args.province] if args.province else list(PROVINCE_PDFS)
    for prov in provinces:
        extract_province(prov, client, skip_ocr=args.skip_ocr, force=args.force)


if __name__ == "__main__":
    main()
