"""
Extract raw layout (x0, y0, font_size, font_name, text) from all 3 province PDFs.
Produces per-province JSON files used by the parsers.
"""
import fitz
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "raw" / "pdf"
OUT_DIR = BASE_DIR / "data" / "extracted" / "raw_layout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVINCE_PDFS = {
    "ON": "moh-schedule-benefit-2025-03-19 (Nov 18, 2025).pdf",
    "BC": "msc_payment_schedule_-_dec_31_2024 (May 29, 2025).pdf",
    "YT": "yukon_physician_fee_guide_2024 (Nov 18, 2025).pdf",
}


def extract_layout(pdf_path: Path, province: str) -> list:
    doc = fitz.open(str(pdf_path))
    all_pages = []

    for page_index, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_data = {"page": page_index + 1, "lines": []}

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                parts, x0, y0, max_size, font_name = [], None, None, None, None
                for span in line["spans"]:
                    txt = span["text"]
                    if txt.strip():
                        parts.append(txt)
                        if x0 is None:
                            x0 = span["bbox"][0]
                            y0 = span["bbox"][1]
                            max_size = span["size"]
                            font_name = span.get("font", "")
                        else:
                            max_size = max(max_size, span["size"])

                text = "".join(parts).strip()
                if not text:
                    continue

                page_data["lines"].append({
                    "text": text,
                    "x0": round(x0, 2) if x0 is not None else None,
                    "y0": round(y0, 2) if y0 is not None else None,
                    "font_size": round(max_size, 2) if max_size is not None else None,
                    "font": font_name or "",
                })

        all_pages.append(page_data)

    doc.close()
    return all_pages


def main():
    for province, pdf_name in PROVINCE_PDFS.items():
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"[SKIP] {pdf_name} not found")
            continue

        out_path = OUT_DIR / f"{province.lower()}_layout.json"
        if out_path.exists():
            print(f"[SKIP] {out_path.name} already exists — delete to re-run")
            continue

        print(f"[{province}] Extracting {pdf_name} ...", flush=True)
        pages = extract_layout(pdf_path, province)
        out_path.write_text(
            json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{province}] Done - {len(pages)} pages -> {out_path}")


if __name__ == "__main__":
    main()
