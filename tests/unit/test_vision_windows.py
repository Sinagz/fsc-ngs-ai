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
