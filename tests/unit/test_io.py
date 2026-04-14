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
    block = blocks[0]
    assert block.x0 > 0, "x0 should reflect actual span position"
    assert block.y0 > 0, "y0 should reflect actual span position"
    assert block.size > 0, "font size should be positive"
    assert block.font != "", "font name should be populated"


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
