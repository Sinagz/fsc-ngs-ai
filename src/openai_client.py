"""Single chokepoint for all OpenAI calls.

No module in src/pipeline or src/core imports openai directly — everything
flows through :class:`OpenAIClient`. The wrapper provides:

* httpx transport layered with a hishel cache so re-running the pipeline
  does not re-spend on OpenAI when request bodies are unchanged.
* Retry with exponential backoff on transient failures (both chat and
  embed paths).
* Per-model token accounting via :class:`CostTracker` for diagnostics.
* Pydantic-validated structured output through
  ``response_format={"type": "json_schema", "strict": True}``.
* Batched embeddings, results sorted by the authoritative ``index`` field.

Hishel 1.x caching strategy
---------------------------
OpenAI calls are all HTTPS POST and OpenAI does not send ``Cache-Control``
headers. Hishel's default :class:`SpecificationPolicy` therefore refuses
to cache such traffic — RFC 9111 Section 3 requires both a cacheable
method (``supported_methods`` defaults to ``["GET", "HEAD"]``) and a
storable response; neither condition is met here.

To make the cache actually work we use
:class:`hishel._policies.FilterPolicy` with an empty filter list and
``use_body_key=True``. That combination:

* bypasses the spec's storage checks (an empty filter chain rejects
  nothing, so every response is stored), and
* keys entries on ``sha256(request_body)`` so two identical OpenAI calls
  — e.g. replaying a pipeline step — are served from the local SQLite
  cache instead of re-billing.

Verified interactively against an ``httpx.MockTransport`` on hishel
1.1.9: two identical POSTs produce one upstream call; a POST with a
different body produces a second.

Hishel 1.x API notes
--------------------
* ``hishel.httpx.SyncCacheTransport(next_transport=..., storage=...,
  policy=...)`` — the kwarg is ``next_transport``, not ``transport``.
* Storage is ``hishel.SyncSqliteStorage(database_path=...)``; there is
  no ``FileStorage`` at the top level anymore.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

import hishel
import httpx
from hishel._policies import FilterPolicy
from hishel.httpx import SyncCacheTransport
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".hishel-cache"


class _Deterministic(Exception):
    """Marker wrapper: the inner error must not be retried."""

    def __init__(self, inner: Exception) -> None:
        super().__init__(str(inner))
        self.inner = inner


# OpenAI strict JSON-schema mode rejects pydantic's default schema output.
# Unsupported keywords per the structured-outputs docs (as of 2026):
#   - numeric/string constraints (minimum, maximum, minLength, pattern, ...)
#   - 'default' on any field
#   - 'additionalProperties' omitted (must be explicitly false)
#   - 'required' missing fields (all properties must be required)
# This helper rewrites a pydantic-generated schema in place to satisfy strict
# mode while preserving semantics. Optional fields (pydantic default=None) stay
# optional via "anyOf: [type, null]" which strict mode *does* allow.
_STRIP_KEYS = frozenset({
    "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern", "minItems", "maxItems",
    "uniqueItems", "format",
})


def _strictify_schema(node: object) -> None:
    """Recursively rewrite a JSON schema so OpenAI strict mode accepts it."""
    if isinstance(node, dict):
        for k in list(node.keys()):
            if k in _STRIP_KEYS:
                node.pop(k)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _strictify_schema(v)
    elif isinstance(node, list):
        for item in node:
            _strictify_schema(item)


@dataclass
class CostTracker:
    """Accumulates token counts per model across a pipeline run.

    Thread-safe: uses a lock to protect read-modify-write operations.
    Safe for concurrent record() calls from multiple threads (Task 9
    uses asyncio.to_thread with Semaphore(20)).
    """

    _by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
    ) -> None:
        with self._lock:
            slot = self._by_model.setdefault(
                model,
                {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
            )
            slot["prompt_tokens"] += prompt_tokens
            slot["completion_tokens"] += completion_tokens
            slot["calls"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a shallow-per-model copy so callers cannot mutate state."""
        with self._lock:
            return {m: dict(s) for m, s in self._by_model.items()}


class OpenAIClient:
    """Cached, retrying wrapper around the OpenAI SDK."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment")

        cache_path = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
        cache_path.mkdir(parents=True, exist_ok=True)

        storage = hishel.SyncSqliteStorage(
            database_path=str(cache_path / "hishel_cache.db"),
        )
        policy = FilterPolicy()
        policy.use_body_key = True

        transport = SyncCacheTransport(
            next_transport=httpx.HTTPTransport(),
            storage=storage,
            policy=policy,
        )
        http_client = httpx.Client(transport=transport, timeout=60.0)

        self._sdk = OpenAI(api_key=key, http_client=http_client)
        self.costs = CostTracker()

    def _with_retry(
        self,
        fn: Callable[[], R],
        *,
        max_retries: int = 3,
    ) -> R:
        """Run ``fn`` with exponential backoff.

        A :class:`_Deterministic` wrapper short-circuits retries and
        re-raises its ``inner`` exception unchanged.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                return fn()
            except _Deterministic as det:
                raise det.inner from None
            except Exception as exc:  # noqa: BLE001 - retried then re-raised
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(
            f"OpenAI call failed after {max_retries} retries: {last_err}"
        )

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
        """Run a chat completion with strict JSON-schema output and validate.

        Deterministic failures (``json.JSONDecodeError``,
        :class:`pydantic.ValidationError`, null content / usage) are raised
        immediately — retrying at ``temperature=0`` would just burn tokens
        for the same result. Transport/API errors are retried with
        exponential backoff.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        raw_schema = schema.model_json_schema()
        _strictify_schema(raw_schema)
        json_schema = {
            "name": schema.__name__,
            "schema": raw_schema,
            "strict": True,
        }

        def _call() -> T:
            resp = self._sdk.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
            )

            finish = (
                resp.choices[0].finish_reason
                if resp.choices
                else "unknown"
            )

            if resp.usage is None:
                raise _Deterministic(
                    ValueError(
                        f"OpenAI returned no usage block "
                        f"(finish_reason={finish})"
                    )
                )

            content = (
                resp.choices[0].message.content if resp.choices else None
            )
            if content is None:
                raise _Deterministic(
                    ValueError(
                        f"OpenAI returned null content "
                        f"(finish_reason={finish})"
                    )
                )

            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise _Deterministic(exc) from exc

            try:
                parsed = schema.model_validate(payload)
            except ValidationError as exc:
                raise _Deterministic(exc) from exc

            self.costs.record(
                model=resp.model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
            return parsed

        return self._with_retry(_call, max_retries=max_retries)

    def chat_vision_json(
        self,
        *,
        prompt: str,
        images: list[bytes],
        schema: type[T],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> T:
        """Vision-enabled chat_json.

        ``images`` is a list of raw image bytes (PNG); each is base64-encoded
        into a ``data:`` URL with ``detail=high``. The request body, including
        image bytes, forms the hishel cache key, so identical inputs replay
        from the SQLite cache.

        Deterministic failures (``json.JSONDecodeError``,
        :class:`pydantic.ValidationError`, null content / usage) are raised
        immediately — retrying at ``temperature=0`` would just burn tokens
        for the same result. Transport/API errors are retried with
        exponential backoff.
        """
        if not images:
            raise ValueError("chat_vision_json requires at least one image")

        import base64

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})

        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        messages.append({"role": "user", "content": content})

        raw_schema = schema.model_json_schema()
        _strictify_schema(raw_schema)
        json_schema = {
            "name": schema.__name__,
            "schema": raw_schema,
            "strict": True,
        }

        def _call() -> T:
            resp = self._sdk.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
            )

            finish = (
                resp.choices[0].finish_reason
                if resp.choices
                else "unknown"
            )

            if resp.usage is None:
                raise _Deterministic(
                    ValueError(
                        f"OpenAI returned no usage block "
                        f"(finish_reason={finish})"
                    )
                )

            content_raw = (
                resp.choices[0].message.content if resp.choices else None
            )
            if content_raw is None:
                raise _Deterministic(
                    ValueError(
                        f"OpenAI returned null content "
                        f"(finish_reason={finish})"
                    )
                )

            try:
                payload = json.loads(content_raw)
            except json.JSONDecodeError as exc:
                raise _Deterministic(exc) from exc

            try:
                parsed = schema.model_validate(payload)
            except ValidationError as exc:
                raise _Deterministic(exc) from exc

            self.costs.record(
                model=resp.model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
            return parsed

        return self._with_retry(_call, max_retries=max_retries)

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dim: int,
        batch_size: int = 500,
        max_retries: int = 3,
    ) -> list[list[float]]:
        """Embed ``texts`` in batches and return vectors in input order.

        Each batch is retried independently with exponential backoff, so a
        single transient 429 mid-run does not discard earlier vectors.
        Results are sorted by ``data[i].index`` because the OpenAI API does
        not guarantee response ordering matches input order.
        """
        out: list[list[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if total_batches > 1:
            iterator = tqdm(
                iterator, total=total_batches, desc="Embeddings",
                unit="batch", leave=False,
            )
        for i in iterator:
            batch = texts[i : i + batch_size]

            def _call(b: list[str] = batch) -> None:
                resp = self._sdk.embeddings.create(
                    input=b,
                    model=model,
                    dimensions=dim,
                )
                self.costs.record(
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                )
                sorted_data = sorted(resp.data, key=lambda d: d.index)
                out.extend(d.embedding for d in sorted_data)

            self._with_retry(_call, max_retries=max_retries)
        return out
