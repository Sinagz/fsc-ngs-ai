from unittest.mock import MagicMock, patch

import pytest

from src.openai_client import CostTracker, OpenAIClient


def test_cost_tracker_accumulates():
    t = CostTracker()
    t.record(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    t.record(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=0)
    snap = t.snapshot()
    assert snap["gpt-4o-mini"]["prompt_tokens"] == 300
    assert snap["gpt-4o-mini"]["completion_tokens"] == 50


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_chat_json_validates_against_schema(monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        code: str
        price: float

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '{"code": "K040", "price": 78.45}'
    fake_resp.usage.prompt_tokens = 50
    fake_resp.usage.completion_tokens = 10
    fake_resp.model = "gpt-4o-mini"

    with patch.object(client._sdk.chat.completions, "create", return_value=fake_resp):
        parsed = client.chat_json(prompt="hi", schema=Out, model="gpt-4o-mini")

    assert parsed == Out(code="K040", price=78.45)
    assert client.costs.snapshot()["gpt-4o-mini"]["prompt_tokens"] == 50


def test_embed_batches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    def fake_create(input, model, dimensions, **kw):
        return MagicMock(
            data=[MagicMock(embedding=[0.1] * dimensions) for _ in input],
            usage=MagicMock(prompt_tokens=len(input), total_tokens=len(input)),
            model=model,
        )

    with patch.object(client._sdk.embeddings, "create", side_effect=fake_create):
        vecs = client.embed(["a"] * 1200, model="text-embedding-3-large", dim=1024)

    assert len(vecs) == 1200
    assert len(vecs[0]) == 1024
