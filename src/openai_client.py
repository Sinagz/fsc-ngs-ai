"""Single chokepoint for all OpenAI calls.

No module in src/pipeline or src/core imports openai directly — everything
flows through :class:`OpenAIClient`. The wrapper provides:

* httpx transport layered with a hishel cache so re-running the pipeline
  does not re-spend on OpenAI when request bodies are unchanged.
* Retry with exponential backoff on transient failures.
* Per-model token accounting via :class:`CostTracker` for diagnostics.
* Pydantic-validated structured output through
  ``response_format={"type": "json_schema", "strict": True}``.
* Batched embeddings.

Hishel 1.x API note
-------------------
Hishel 1.1.9 exposes its httpx integration via
``hishel.httpx.SyncCacheTransport`` (kwarg ``next_transport``) and
``hishel.SyncSqliteStorage`` (kwarg ``database_path``). There is no
``FileStorage`` / ``CacheTransport`` at the top level anymore, so the
cache backend here is SQLite-on-disk rather than per-file on-disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import hishel
import httpx
from hishel.httpx import SyncCacheTransport
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class CostTracker:
    """Accumulates token counts per model across a pipeline run."""

    _by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
    ) -> None:
        slot = self._by_model.setdefault(
            model,
            {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        )
        slot["prompt_tokens"] += prompt_tokens
        slot["completion_tokens"] += completion_tokens
        slot["calls"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a deep-ish copy so callers cannot mutate internal state."""
        return {m: dict(s) for m, s in self._by_model.items()}


class OpenAIClient:
    """Cached, retrying wrapper around the OpenAI SDK."""

    def __init__(self, cache_dir: str = ".hishel-cache") -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        storage = hishel.SyncSqliteStorage(
            database_path=str(cache_path / "hishel_cache.db"),
        )
        transport = SyncCacheTransport(
            next_transport=httpx.HTTPTransport(),
            storage=storage,
        )
        http_client = httpx.Client(transport=transport, timeout=60.0)

        self._sdk = OpenAI(api_key=key, http_client=http_client)
        self.costs = CostTracker()

    def chat_json(
        self,
        *,
        prompt: str,
        schema: type[T],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> T:
        """Run a chat completion with strict JSON-schema output and validate."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        json_schema = {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        }

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._sdk.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema,
                    },
                )
                self.costs.record(
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
                return schema.model_validate(
                    json.loads(resp.choices[0].message.content)
                )
            except Exception as exc:  # noqa: BLE001 - retried then re-raised
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            f"chat_json failed after {max_retries} retries: {last_err}"
        )

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dim: int,
        batch_size: int = 500,
    ) -> list[list[float]]:
        """Embed ``texts`` in batches and return a list of vectors."""
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._sdk.embeddings.create(
                input=batch,
                model=model,
                dimensions=dim,
            )
            self.costs.record(
                model=resp.model,
                prompt_tokens=resp.usage.prompt_tokens,
            )
            out.extend(d.embedding for d in resp.data)
        return out
