"""Hypothesis property tests for vision window building."""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from src.pipeline.vision.toc import SectionContext
from src.pipeline.vision.windows import build_windows


@pytest.mark.property
@settings(max_examples=50)
@given(st.integers(min_value=1, max_value=500))
def test_n_pages_yields_n_windows(n: int) -> None:
    """build_windows yields exactly n windows for n pages."""
    windows = list(build_windows(num_pages=n, section_map={}))
    assert len(windows) == n


@pytest.mark.property
@settings(max_examples=50)
@given(st.integers(min_value=1, max_value=500))
def test_every_page_is_target_exactly_once(n: int) -> None:
    """Every page number 1..n appears as target_page exactly once."""
    windows = list(build_windows(num_pages=n, section_map={}))
    targets = [w.target_page for w in windows]
    assert sorted(targets) == list(range(1, n + 1))


@pytest.mark.property
@settings(max_examples=50)
@given(st.integers(min_value=2, max_value=500))
def test_context_page_is_previous_except_first(n: int) -> None:
    """context_page is None for window 0; page-1 for windows 1..n-1."""
    windows = list(build_windows(num_pages=n, section_map={}))
    assert windows[0].context_page is None
    for w in windows[1:]:
        assert w.context_page == w.target_page - 1


@pytest.mark.property
@settings(max_examples=50)
@given(st.integers(min_value=1, max_value=100))
def test_section_hints_default_to_empty_when_no_map(n: int) -> None:
    """section_hints defaults to empty SectionContext when section_map is empty."""
    windows = list(build_windows(num_pages=n, section_map={}))
    empty = SectionContext(None, None, None)
    for w in windows:
        assert w.section_hints == empty
