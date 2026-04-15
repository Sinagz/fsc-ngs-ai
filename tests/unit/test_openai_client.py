from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.openai_client import CostTracker, OpenAIClient


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


def test_cost_tracker_accumulates():
    t = CostTracker()
    t.record(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    t.record(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=0)
    snap = t.snapshot()
    assert snap["gpt-4o-mini"]["prompt_tokens"] == 300
    assert snap["gpt-4o-mini"]["completion_tokens"] == 50


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient()


# ---------------------------------------------------------------------------
# chat_json
# ---------------------------------------------------------------------------


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


def test_chat_json_raises_on_null_content_without_retry(monkeypatch):
    """A None ``message.content`` (e.g. content_filter refusal) must raise
    :class:`ValueError` immediately — no retry, no misleading wrapping."""
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = None
    fake_resp.choices[0].finish_reason = "content_filter"
    fake_resp.usage.prompt_tokens = 5
    fake_resp.usage.completion_tokens = 0
    fake_resp.model = "gpt-4o-mini"

    with patch.object(
        client._sdk.chat.completions, "create", return_value=fake_resp
    ) as mock_create:
        with pytest.raises(ValueError, match="null content"):
            client.chat_json(prompt="hi", schema=Out, model="gpt-4o-mini")

    assert mock_create.call_count == 1


def test_chat_json_raises_on_null_usage_without_retry(monkeypatch):
    """A None ``resp.usage`` must raise :class:`ValueError` immediately."""
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '{"x": 1}'
    fake_resp.choices[0].finish_reason = "length"
    fake_resp.usage = None
    fake_resp.model = "gpt-4o-mini"

    with patch.object(
        client._sdk.chat.completions, "create", return_value=fake_resp
    ) as mock_create:
        with pytest.raises(ValueError, match="no usage block"):
            client.chat_json(prompt="hi", schema=Out, model="gpt-4o-mini")

    assert mock_create.call_count == 1


def test_chat_json_fast_fails_on_validation_error(monkeypatch):
    """Pydantic ValidationError is deterministic at temp=0 — no retry."""
    from pydantic import BaseModel, ValidationError

    class Out(BaseModel):
        code: str
        price: float

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '{"code": "K040"}'  # missing price
    fake_resp.choices[0].finish_reason = "stop"
    fake_resp.usage.prompt_tokens = 5
    fake_resp.usage.completion_tokens = 5
    fake_resp.model = "gpt-4o-mini"

    with patch.object(
        client._sdk.chat.completions, "create", return_value=fake_resp
    ) as mock_create:
        with pytest.raises(ValidationError):
            client.chat_json(prompt="hi", schema=Out, model="gpt-4o-mini")

    assert mock_create.call_count == 1


def test_chat_json_retries_transient_errors(monkeypatch):
    """Transport-level errors should be retried."""
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Suppress backoff sleeps to keep the test fast.
    monkeypatch.setattr("src.openai_client.time.sleep", lambda _s: None)

    client = OpenAIClient()

    fake_ok = MagicMock()
    fake_ok.choices = [MagicMock()]
    fake_ok.choices[0].message.content = '{"ok": true}'
    fake_ok.choices[0].finish_reason = "stop"
    fake_ok.usage.prompt_tokens = 1
    fake_ok.usage.completion_tokens = 1
    fake_ok.model = "gpt-4o-mini"

    side_effects = [RuntimeError("transient"), fake_ok]
    with patch.object(
        client._sdk.chat.completions, "create", side_effect=side_effects
    ) as mock_create:
        parsed = client.chat_json(prompt="hi", schema=Out, model="gpt-4o-mini")

    assert parsed == Out(ok=True)
    assert mock_create.call_count == 2


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


def test_embed_batches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    def fake_create(input, model, dimensions, **kw):
        return MagicMock(
            data=[
                MagicMock(embedding=[0.1] * dimensions, index=i)
                for i in range(len(input))
            ],
            usage=MagicMock(prompt_tokens=len(input), total_tokens=len(input)),
            model=model,
        )

    with patch.object(client._sdk.embeddings, "create", side_effect=fake_create):
        vecs = client.embed(["a"] * 1200, model="text-embedding-3-large", dim=1024)

    assert len(vecs) == 1200
    assert len(vecs[0]) == 1024


def test_embed_preserves_input_order_via_index(monkeypatch):
    """OpenAI does not guarantee ``resp.data`` ordering — embed() must sort
    by ``d.index`` so per-text vectors line up with the input list."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient()

    def fake_create(input, model, dimensions, **kw):
        # Encode input position into the first component of each vector.
        entries = []
        for i, text in enumerate(input):
            vec = [0.0] * dimensions
            vec[0] = float(text)  # text == str(position)
            entries.append(MagicMock(embedding=vec, index=i))
        # Deliberately return data in reversed order to stress ordering.
        entries.reverse()
        return MagicMock(
            data=entries,
            usage=MagicMock(prompt_tokens=len(input), total_tokens=len(input)),
            model=model,
        )

    texts = [str(i) for i in range(1200)]
    with patch.object(client._sdk.embeddings, "create", side_effect=fake_create):
        vecs = client.embed(texts, model="text-embedding-3-large", dim=16)

    # vecs[i] must correspond to texts[i], regardless of API response order.
    # The fake encoded float(text) into slot 0, and texts[i] == str(i), so
    # the expected value is simply float(i).
    for i, v in enumerate(vecs):
        assert v[0] == float(i), (
            f"mismatch at index {i}: got {v[0]} want {float(i)}"
        )


def test_embed_retries_transient_errors(monkeypatch):
    """embed() must retry on transient failures (not just chat_json)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("src.openai_client.time.sleep", lambda _s: None)
    client = OpenAIClient()

    def fake_ok(input, model, dimensions, **kw):
        return MagicMock(
            data=[
                MagicMock(embedding=[0.5] * dimensions, index=i)
                for i in range(len(input))
            ],
            usage=MagicMock(prompt_tokens=len(input), total_tokens=len(input)),
            model=model,
        )

    side_effects = [
        RuntimeError("429 rate limited"),
        fake_ok(["a", "b"], "m", 4),
    ]
    with patch.object(
        client._sdk.embeddings, "create", side_effect=side_effects
    ) as mock_create:
        vecs = client.embed(
            ["a", "b"], model="text-embedding-3-large", dim=4
        )

    assert mock_create.call_count == 2
    assert len(vecs) == 2
    assert vecs[0] == [0.5] * 4


# ---------------------------------------------------------------------------
# Hishel cache integration (the critical fix)
# ---------------------------------------------------------------------------


def test_hishel_cache_serves_identical_post_from_disk(monkeypatch):
    """Regression guard for the POST-caching fix.

    OpenAI traffic is HTTPS POST with no ``Cache-Control`` headers, so
    hishel's default :class:`SpecificationPolicy` would silently refuse
    to cache. :class:`OpenAIClient` must configure a
    :class:`FilterPolicy` with ``use_body_key=True`` so identical request
    bodies are served from the on-disk SQLite cache.

    This test wires the real ``SyncCacheTransport`` (as configured by
    :class:`OpenAIClient`) underneath a counting
    :class:`httpx.MockTransport` and asserts two identical POSTs produce
    one upstream call, while a POST with a different body produces a
    second.
    """
    import hishel
    from hishel._policies import FilterPolicy
    from hishel.httpx import SyncCacheTransport

    call_count = {"n": 0}

    def origin(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"ok": True})

    mock_origin = httpx.MockTransport(origin)

    with tempfile.TemporaryDirectory() as tmp:
        storage = hishel.SyncSqliteStorage(
            database_path=str(Path(tmp) / "cache.db")
        )
        policy = FilterPolicy()
        policy.use_body_key = True
        cache_t = SyncCacheTransport(
            next_transport=mock_origin,
            storage=storage,
            policy=policy,
        )
        with httpx.Client(transport=cache_t) as http:
            r1 = http.post("https://api.example/v1/x", json={"a": 1})
            r2 = http.post("https://api.example/v1/x", json={"a": 1})
            r3 = http.post("https://api.example/v1/x", json={"a": 2})

    assert r1.status_code == r2.status_code == r3.status_code == 200
    # Two identical payloads collapse into one origin call; the third
    # differs and costs a second call.
    assert call_count["n"] == 2, (
        "hishel cache is not active for POST — "
        "FilterPolicy(use_body_key=True) must be configured"
    )


def test_openai_client_uses_filter_policy(monkeypatch, tmp_path):
    """Structural guard: :class:`OpenAIClient` must wire a
    :class:`FilterPolicy` with ``use_body_key=True`` (not the default
    :class:`SpecificationPolicy`, which would skip POST caching)."""
    from hishel._policies import FilterPolicy

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient(cache_dir=tmp_path)

    # Reach through the SDK to inspect the transport we installed.
    transport = client._sdk._client._transport  # type: ignore[attr-defined]
    assert hasattr(transport, "_cache_proxy"), (
        "OpenAIClient did not install a SyncCacheTransport"
    )
    policy = transport._cache_proxy.policy
    assert isinstance(policy, FilterPolicy), (
        f"Expected FilterPolicy to cache POST; got {type(policy).__name__}"
    )
    assert policy.use_body_key is True, (
        "FilterPolicy must key entries on request body hash"
    )


def test_cost_tracker_thread_safe_record():
    """CostTracker.record must be thread-safe under asyncio.to_thread
    concurrency (Task 9 uses Semaphore(20) concurrent threads).

    Verify that 10 threads each calling record() 100 times accumulates
    correctly without lost counts due to non-atomic read-modify-write.
    """
    import threading

    tracker = CostTracker()
    n_threads = 10
    per_thread = 100
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()  # start all threads as simultaneously as possible
        for _ in range(per_thread):
            tracker.record(model="gpt-test", prompt_tokens=1, completion_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = tracker.snapshot()
    assert snap["gpt-test"]["prompt_tokens"] == n_threads * per_thread
    assert snap["gpt-test"]["completion_tokens"] == n_threads * per_thread
    assert snap["gpt-test"]["calls"] == n_threads * per_thread


def test_openai_client_concurrent_cache_writes(monkeypatch, tmp_path):
    """20 threads writing to the same hishel cache must not deadlock or lose entries."""
    import json as _json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import MagicMock, patch

    from pydantic import BaseModel, ConfigDict

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIClient(cache_dir=tmp_path / "cache")

    class _Tiny(BaseModel):
        model_config = ConfigDict(extra="forbid")
        code: str

    # Return a unique response per call so each request is a cache MISS and writes.
    call_counter = {"n": 0}
    lock = threading.Lock()

    def _fake_create(**kwargs):
        with lock:
            call_counter["n"] += 1
            n = call_counter["n"]
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = _json.dumps({"code": f"R{n:03d}"})
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.model = "gpt-test"
        return resp

    with patch.object(client._sdk.chat.completions, "create", side_effect=_fake_create):
        def _one(i: int):
            return client.chat_json(
                prompt=f"request-{i}",  # distinct body -> distinct cache key
                schema=_Tiny,
                model="gpt-test",
            )

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(_one, range(20)))

    assert len(results) == 20
    # Verify WAL mode is configured on the underlying SQLite DB.
    # (The mock intercepts above the transport layer so hishel never writes
    # cache entries here; the DB file is created by _make_cache_connection.)
    import sqlite3
    db_path = tmp_path / "cache" / "hishel_cache.db"
    assert db_path.exists(), "hishel cache DB was not created"
    conn = sqlite3.connect(str(db_path))
    journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode == "wal", (
        f"Expected WAL journal mode for concurrent writes; got '{journal_mode}'"
    )
    conn.close()
