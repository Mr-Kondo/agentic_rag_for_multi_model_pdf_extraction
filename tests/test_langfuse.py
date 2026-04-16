"""Unit tests for the Langfuse tracing wrapper."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from src.integrations.langfuse import LangfuseTracer, TraceHandle, _build_usage_details


class _FakeGeneration:
    def __init__(self, fail_on_update: bool = False):
        self.fail_on_update = fail_on_update
        self.updates: list[dict] = []

    def update(self, **kwargs):
        if self.fail_on_update:
            raise RuntimeError("update failed")
        self.updates.append(kwargs)


class _FakeSpan:
    def __init__(self, generation: _FakeGeneration):
        self.generation = generation
        self.calls: list[dict] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.calls.append(kwargs)
        yield self.generation


class _FakeTraceContext:
    def __init__(self):
        self.child_generation = _FakeGeneration()
        self.generation_calls: list[dict] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.generation_calls.append(kwargs)
        yield self.child_generation


class _FallbackTraceContext(_FakeTraceContext):
    @contextmanager
    def start_observation(self, **kwargs):
        self.generation_calls.append(kwargs)
        yield self.child_generation


class _FakeClientWithFallbackTraceApi:
    def __init__(self):
        self.calls: list[dict] = []
        self.trace_context = _FallbackTraceContext()

    def start_observation(self, **kwargs):
        self.calls.append(kwargs)
        return self.trace_context

    def get_current_trace_id(self):
        return "trace-fallback-123"

    def create_score(self, **kwargs):
        return kwargs


class _FakeClientMissingTraceApi:
    def create_score(self, **kwargs):
        return kwargs


class _FakeClientWithLegacyScore:
    def __init__(self):
        self.score_calls: list[dict] = []

    def score(self, **kwargs):
        self.score_calls.append(kwargs)


def test_build_usage_details_returns_langfuse_usage_shape():
    """Usage payload should use Langfuse's native input/output/total token keys."""
    assert _build_usage_details(11, 7) == {
        "input": 11,
        "output": 7,
        "total": 18,
    }
    assert _build_usage_details(None, 7) is None
    assert _build_usage_details(11, None) is None


def test_trace_generation_updates_usage_details_even_with_zero_tokens():
    """Zero token counts should be propagated immediately via set_output(), not dropped."""
    generation = _FakeGeneration()
    raw_span = _FakeSpan(generation)
    trace = TraceHandle(raw_span, trace_id="trace-123")

    with trace.generation(name="test_generation", model="test-model", input={"messages": []}) as handle:
        # set_output() must be called inside the context manager (span is alive here)
        handle.set_output("ok", input_tokens=0, output_tokens=5)

    assert raw_span.calls[0]["name"] == "test_generation"
    assert generation.updates == [
        {
            "output": "ok",
            "model": "test-model",
            "usage_details": {
                "input": 0,
                "output": 5,
                "total": 5,
            },
        }
    ]


def test_trace_generation_logs_update_failures(caplog):
    """Langfuse update failures should be surfaced in logs without crashing the caller."""
    generation = _FakeGeneration(fail_on_update=True)
    raw_span = _FakeSpan(generation)
    trace = TraceHandle(raw_span, trace_id="trace-123")

    with caplog.at_level(logging.WARNING):
        with trace.generation(name="test_generation", model="test-model") as handle:
            handle.set_output("oops", input_tokens=3, output_tokens=2)

    assert "Failed to update generation with output/usage" in caplog.text


def test_tracer_uses_fallback_trace_api(monkeypatch):
    """Tracer should fall back to start_span when start_as_current_span is unavailable."""
    fake_client = _FakeClientWithFallbackTraceApi()
    monkeypatch.setattr("src.integrations.langfuse._get_client", lambda: fake_client)

    tracer = LangfuseTracer()

    with tracer.trace("rag_query", input={"question": "hi"}) as trace:
        assert trace.trace_id == "trace-fallback-123"
        with trace.generation(name="orchestrator", model="test-model") as handle:
            handle.set_output("ok")

    assert fake_client.calls[0]["name"] == "rag_query"
    assert fake_client.trace_context.generation_calls[0]["name"] == "orchestrator"


def test_tracer_logs_reason_when_trace_api_is_missing(monkeypatch, caplog):
    """Missing client trace APIs should produce a reasoned no-op diagnostic."""
    monkeypatch.setattr("src.integrations.langfuse._get_client", lambda: _FakeClientMissingTraceApi())

    tracer = LangfuseTracer()

    with caplog.at_level(logging.WARNING):
        with tracer.trace("rag_query") as trace:
            assert trace.trace_id == "no-op"
            assert trace.disable_reason == "missing_client_trace_api"

    assert "Langfuse trace unavailable." in caplog.text
    assert "reason=missing_client_trace_api" in caplog.text


def test_trace_generation_logs_reason_when_generation_api_is_missing(caplog):
    """Missing generation APIs on an active trace should degrade to no-op with diagnostics."""
    trace = TraceHandle(raw=object())

    with caplog.at_level(logging.WARNING):
        with trace.generation(name="test_generation", model="test-model") as handle:
            handle.set_output("ignored")

    assert "Langfuse generation unavailable." in caplog.text
    assert "reason=missing_generation_api" in caplog.text


def test_score_falls_back_to_legacy_score_api(monkeypatch):
    """Score posting should use legacy score() when create_score() is unavailable."""
    fake_client = _FakeClientWithLegacyScore()
    monkeypatch.setattr("src.integrations.langfuse._get_client", lambda: fake_client)

    tracer = LangfuseTracer()
    tracer.score(trace_id="trace-123", name="answer_grounding", value=0.9)

    assert fake_client.score_calls == [
        {
            "trace_id": "trace-123",
            "name": "answer_grounding",
            "value": 0.9,
            "comment": None,
            "data_type": "NUMERIC",
        }
    ]
