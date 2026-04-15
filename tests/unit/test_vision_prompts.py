from src.pipeline.vision.prompts import SYSTEM_TEMPLATE, build_prompt
from src.pipeline.vision.toc import SectionContext


def test_system_template_mentions_target_page_rule():
    assert "TARGET PAGE" in SYSTEM_TEMPLATE
    assert "final visible line" in SYSTEM_TEMPLATE


def test_build_prompt_includes_province_and_target_page():
    hints = SectionContext(chapter=None, section=None, subsection=None)
    prompt = build_prompt(province="ON", target_page=47, section=hints)
    assert "ON" in prompt
    assert "page 47" in prompt or "page=47" in prompt


def test_build_prompt_injects_toc_hints_when_present():
    hints = SectionContext(
        chapter="Diagnostic Imaging",
        section="X-Ray",
        subsection=None,
    )
    prompt = build_prompt(province="ON", target_page=120, section=hints)
    assert "Diagnostic Imaging" in prompt
    assert "X-Ray" in prompt


def test_build_prompt_says_null_when_no_hints():
    hints = SectionContext(chapter=None, section=None, subsection=None)
    prompt = build_prompt(province="YT", target_page=12, section=hints)
    assert "null" in prompt.lower()
