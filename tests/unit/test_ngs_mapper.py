from decimal import Decimal
from unittest.mock import MagicMock

from src.pipeline.schema import FeeCodeRecord, NGSRecord
from src.pipeline.ngs_mapper import map_ngs, NGSVerdict


def _fee(code="K040"):
    return FeeCodeRecord(
        province="ON", fsc_code=code, fsc_fn="fn", fsc_description="Periodic visit",
        page=1, source_pdf_hash="a" * 64,
        extraction_method="structural", extraction_confidence=1.0,
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
