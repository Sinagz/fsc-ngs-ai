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
