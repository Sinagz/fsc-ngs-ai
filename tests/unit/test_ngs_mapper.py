from decimal import Decimal
from unittest.mock import MagicMock

from src.pipeline.schema import FeeCodeRecord, NGSRecord
from src.pipeline.ngs_mapper import map_ngs, NGSVerdict


def _fee(code="K040"):
    return FeeCodeRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="Periodic visit",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="vision", extraction_confidence=1.0,
    )


def _ngs(code="1AA", refs=("K040",)):
    return NGSRecord(ngs_code=code, ngs_label="Exam",
                     ngs_description="Periodic visit", code_refs=list(refs))


def test_exact_match_sets_method_exact():
    client = MagicMock()
    records = map_ngs([_fee()], [_ngs(refs=("K040",))], client=client,
                      embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code == "1AA"
    assert records[0].NGS_mapping_method == "exact"
    assert records[0].NGS_mapping_confidence == 1.0
    client.chat_json.assert_not_called()


def test_no_match_sets_nomap():
    client = MagicMock()
    client.embed.return_value = [[0.0] * 16, [0.0] * 16]
    records = map_ngs([_fee(code="Z999")], [_ngs(refs=())],
                      client=client, embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code is None
    assert records[0].NGS_mapping_method is None
    assert records[0].NGS_mapping_confidence == 0.0


def test_llm_verifies_semantic_candidate():
    client = MagicMock()
    # code not in any NGS code_refs, but semantically matches
    client.embed.return_value = [
        [1.0] + [0.0] * 15,  # fee description embedding
        [1.0] + [0.0] * 15,  # ngs description embedding — cosine = 1.0
    ]
    client.chat_json.return_value = NGSVerdict(accept=True, confidence=0.9)
    records = map_ngs([_fee(code="K999")], [_ngs(refs=())],
                      client=client, embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code == "1AA"
    assert records[0].NGS_mapping_method == "llm"


def test_llm_rejection_falls_back_to_nomap():
    """When LLM verdict is accept=False, the row should be NOMAP, not mapped."""
    client = MagicMock()
    client.embed.return_value = [
        [1.0] + [0.0] * 15,
        [1.0] + [0.0] * 15,
    ]
    client.chat_json.return_value = NGSVerdict(accept=False, confidence=0.0)
    records = map_ngs([_fee(code="K999")], [_ngs(refs=())],
                      client=client, embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert records[0].NGS_code is None
    assert records[0].NGS_mapping_method is None


def test_input_order_preserved_across_mixed_outcomes():
    """Input order must be preserved when exact and unresolved rows are interleaved."""
    client = MagicMock()
    client.embed.return_value = [[0.0] * 16, [0.0] * 16]  # cosine = 0 < threshold → NOMAP
    fees = [
        _fee(code="K040"),   # will exact-match 1AA
        _fee(code="K999"),   # unresolved → NOMAP (similarity 0)
        _fee(code="K040"),   # will also exact-match 1AA (same code twice)
    ]
    records = map_ngs(fees, [_ngs(code="1AA", refs=("K040",))], client=client,
                      embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert len(records) == 3
    assert records[0].fsc_code == "K040"
    assert records[0].NGS_mapping_method == "exact"
    assert records[1].fsc_code == "K999"
    assert records[1].NGS_mapping_method is None
    assert records[2].fsc_code == "K040"
    assert records[2].NGS_mapping_method == "exact"


def test_client_exception_on_chat_json_falls_back_to_nomap():
    """If chat_json raises after embedding succeeds, the row is NOMAP, batch continues."""
    client = MagicMock()
    client.embed.return_value = [
        [1.0] + [0.0] * 15,
        [1.0] + [0.0] * 15,
    ]
    client.chat_json.side_effect = RuntimeError("retries exhausted")
    records = map_ngs([_fee(code="K999")], [_ngs(refs=())],
                      client=client, embed_model="m", llm_model="gpt-4o-mini", dim=16)
    assert len(records) == 1
    assert records[0].NGS_code is None
    assert records[0].NGS_mapping_method is None
