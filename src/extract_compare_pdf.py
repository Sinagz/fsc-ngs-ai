from pathlib import Path
import json
import traceback

PDF_NAME = "moh-schedule-benefit-2025-03-19 (Nov 18, 2025).pdf"

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "raw" / "pdf" / PDF_NAME

OUT_DIR = BASE_DIR / "data" / "extracted"
PYMUPDF4LLM_DIR = OUT_DIR / "pymupdf4llm"
MARKITDOWN_DIR = OUT_DIR / "markitdown"
MARKER_DIR = OUT_DIR / "marker"
RAW_LAYOUT_DIR = OUT_DIR / "raw_layout"

for d in [PYMUPDF4LLM_DIR, MARKITDOWN_DIR, MARKER_DIR, RAW_LAYOUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def extract_with_pymupdf4llm():
    try:
        import pymupdf4llm

        md_text = pymupdf4llm.to_markdown(str(PDF_PATH))
        save_text(PYMUPDF4LLM_DIR / "output.md", md_text)
        print("PyMuPDF4LLM extraction complete.")
    except Exception as e:
        save_text(PYMUPDF4LLM_DIR / "error.txt", traceback.format_exc())
        print(f"PyMuPDF4LLM failed: {e}")


def extract_with_markitdown():
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(str(PDF_PATH))
        text = result.text_content if hasattr(result, "text_content") else str(result)
        save_text(MARKITDOWN_DIR / "output.md", text)
        print("MarkItDown extraction complete.")
    except Exception as e:
        save_text(MARKITDOWN_DIR / "error.txt", traceback.format_exc())
        print(f"MarkItDown failed: {e}")


def extract_with_marker():
    """
    Marker CLI/package behavior can vary by version.
    First step: try importing a conversion function.
    If your local install differs, we will adjust after the first test.
    """
    try:
        # This may vary depending on installed marker version
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        rendered = converter(str(PDF_PATH))
        text = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)
        save_text(MARKER_DIR / "output.md", text)
        print("Marker extraction complete.")
    except Exception as e:
        save_text(MARKER_DIR / "error.txt", traceback.format_exc())
        print(f"Marker failed: {e}")


def extract_raw_layout():
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(PDF_PATH))
        all_pages = []
        readable_lines = []

        for page_index, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_data = {
                "page": page_index + 1,
                "lines": []
            }

            readable_lines.append(f"\n=== PAGE {page_index + 1} ===\n")

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_text_parts = []
                    x0 = None
                    y0 = None
                    max_size = None

                    for span in line["spans"]:
                        txt = span["text"]
                        if txt.strip():
                            line_text_parts.append(txt)
                            if x0 is None:
                                x0 = span["bbox"][0]
                                y0 = span["bbox"][1]
                                max_size = span["size"]
                            else:
                                max_size = max(max_size, span["size"])

                    text = "".join(line_text_parts).strip()
                    if not text:
                        continue

                    line_record = {
                        "text": text,
                        "x0": round(x0, 2) if x0 is not None else None,
                        "y0": round(y0, 2) if y0 is not None else None,
                        "font_size": round(max_size, 2) if max_size is not None else None,
                    }
                    page_data["lines"].append(line_record)

                    readable_lines.append(
                        f"x0={line_record['x0']:>7} | y0={line_record['y0']:>7} | "
                        f"size={line_record['font_size']:>5} | {line_record['text']}"
                    )

            all_pages.append(page_data)

        (RAW_LAYOUT_DIR / "layout_lines.json").write_text(
            json.dumps(all_pages, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        save_text(RAW_LAYOUT_DIR / "layout_lines.txt", "\n".join(readable_lines))
        print("Raw PyMuPDF layout extraction complete.")
    except Exception as e:
        save_text(RAW_LAYOUT_DIR / "error.txt", traceback.format_exc())
        print(f"Raw layout extraction failed: {e}")


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"Using PDF: {PDF_PATH}")
    extract_with_pymupdf4llm()
    extract_with_markitdown()
    extract_with_marker()
    extract_raw_layout()
    print("Done.")


if __name__ == "__main__":
    main()