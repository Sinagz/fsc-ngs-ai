from decimal import Decimal
from unittest.mock import MagicMock

from src.pipeline.schema import CandidateRow
from src.pipeline.semantic import rescue, RescueOutput


def _low_conf_row():
    return CandidateRow(
        province="ON", fsc_code="K040", fsc_fn="", fsc_description="",
        page=47, source_pdf_hash="a" * 64, confidence=0.6,
    )


def test_rescue_skips_rows_above_threshold():
    row = _low_conf_row().model_copy(update={"confidence": 0.9})
    client = MagicMock()
    rescued, unresolved = rescue([row], client=client, model="gpt-4o-mini",
                                 context_lines={}, threshold=0.8)
    assert rescued == [row]
    assert unresolved == []
    client.chat_json.assert_not_called()


def test_rescue_calls_llm_for_low_confidence():
    row = _low_conf_row()
    client = MagicMock()
    client.chat_json.return_value = RescueOutput(
        fsc_fn="Periodic visit", fsc_description="General periodic visit, adult.",
        price="78.45", confidence=0.9, resolved=True,
    )
    rescued, unresolved = rescue([row], client=client, model="gpt-4o-mini",
                                 context_lines={(47, "K040"): "surrounding text"},
                                 threshold=0.8)
    assert len(rescued) == 1
    assert rescued[0].fsc_description == "General periodic visit, adult."
    assert rescued[0].price == Decimal("78.45")
    assert unresolved == []


def test_rescue_logs_unresolved():
    row = _low_conf_row()
    client = MagicMock()
    client.chat_json.return_value = RescueOutput(
        fsc_fn="", fsc_description="", price=None, confidence=0.0, resolved=False,
    )
    rescued, unresolved = rescue([row], client=client, model="gpt-4o-mini",
                                 context_lines={(47, "K040"): "some text"}, threshold=0.8)
    assert rescued == []
    assert unresolved == [row]


def test_rescue_handles_malformed_price():
    """LLM returns garbage for price -> fall back to None, don't crash."""
    row = _low_conf_row()
    client = MagicMock()
    client.chat_json.return_value = RescueOutput(
        fsc_fn="Visit", fsc_description="Some visit.",
        price="not-a-number", confidence=0.85, resolved=True,
    )
    rescued, _ = rescue([row], client=client, model="gpt-4o-mini",
                        context_lines={(47, "K040"): "text"}, threshold=0.8)
    assert len(rescued) == 1
    assert rescued[0].price is None


def test_rescue_handles_client_exception():
    """chat_json raises -> row routed to unresolved, no re-raise."""
    row = _low_conf_row()
    client = MagicMock()
    client.chat_json.side_effect = RuntimeError("retries exhausted")
    rescued, unresolved = rescue(
        [row], client=client, model="gpt-4o-mini",
        context_lines={(47, "K040"): "text"}, threshold=0.8,
    )
    assert rescued == []
    assert unresolved == [row]


def test_rescue_empty_context_skips_llm_call():
    """No context for a row -> route directly to unresolved, no API call."""
    row = _low_conf_row()
    client = MagicMock()
    rescued, unresolved = rescue(
        [row], client=client, model="gpt-4o-mini",
        context_lines={}, threshold=0.8,
    )
    assert rescued == []
    assert unresolved == [row]
    client.chat_json.assert_not_called()


def test_rescue_preserves_structural_when_llm_returns_empty():
    """LLM resolved=True but empty fields -> keep structural values."""
    row = CandidateRow(
        province="ON", fsc_code="K040", fsc_fn="PartialFn",
        fsc_description="Partial desc", page=47,
        source_pdf_hash="a" * 64, confidence=0.6,
    )
    client = MagicMock()
    client.chat_json.return_value = RescueOutput(
        fsc_fn="", fsc_description="", price=None,
        confidence=0.7, resolved=True,
    )
    rescued, _ = rescue([row], client=client, model="gpt-4o-mini",
                        context_lines={(47, "K040"): "text"}, threshold=0.8)
    assert rescued[0].fsc_fn == "PartialFn"
    assert rescued[0].fsc_description == "Partial desc"
