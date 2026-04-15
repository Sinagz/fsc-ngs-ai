"""Tests chat_vision_json by patching the SDK's create() method.

Follows the same mocking pattern as the existing tests/unit/test_openai_client.py
(monkeypatch OPENAI_API_KEY + patch.object on client._sdk.chat.completions.create).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.openai_client import OpenAIClient


class _Tiny(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    score: float = Field(ge=0.0, le=1.0)


def _fake_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 20
    resp.model = "gpt-5.4-mini"
    return resp


def test_chat_vision_json_builds_image_url_parts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    img_a = b"\x89PNG\r\n\x1a\n" + b"AAAA"
    img_b = b"\x89PNG\r\n\x1a\n" + b"BBBB"
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_response({"label": "ok", "score": 0.9})

    with patch.object(
        client._sdk.chat.completions, "create", side_effect=_capture
    ):
        result = client.chat_vision_json(
            prompt="test prompt",
            images=[img_a, img_b],
            schema=_Tiny,
            model="gpt-5.4-mini",
        )

    assert result.label == "ok"
    assert result.score == 0.9
    user_msg = next(m for m in captured["messages"] if m["role"] == "user")
    assert isinstance(user_msg["content"], list)
    assert sum(1 for p in user_msg["content"] if p["type"] == "text") == 1
    assert sum(1 for p in user_msg["content"] if p["type"] == "image_url") == 2
    # Each image becomes a data: URL
    for part in user_msg["content"]:
        if part["type"] == "image_url":
            assert part["image_url"]["url"].startswith("data:image/png;base64,")
            assert part["image_url"]["detail"] == "high"


def test_chat_vision_json_validation_error_does_not_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake = _fake_response({"label": 123})  # wrong type -> ValidationError

    with patch.object(
        client._sdk.chat.completions, "create", return_value=fake
    ) as mock_create:
        with pytest.raises(ValidationError):
            client.chat_vision_json(
                prompt="x",
                images=[b"\x89PNG\r\n\x1a\nimg"],
                schema=_Tiny,
                model="gpt-5.4-mini",
            )
        assert mock_create.call_count == 1  # deterministic error, no retry


def test_chat_vision_json_rejects_empty_images(monkeypatch):
    """chat_vision_json must reject empty images list with clear error.

    An empty images list would silently produce a text-only request to the
    vision model, which is almost certainly a caller bug. Fail loudly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()
    with pytest.raises(ValueError, match="at least one image"):
        client.chat_vision_json(
            prompt="x",
            images=[],
            schema=_Tiny,
            model="gpt-5.4-mini",
        )
