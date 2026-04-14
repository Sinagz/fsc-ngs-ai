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
                                 context_lines={(47, "K040"): ""}, threshold=0.8)
    assert rescued == []
    assert unresolved == [row]
