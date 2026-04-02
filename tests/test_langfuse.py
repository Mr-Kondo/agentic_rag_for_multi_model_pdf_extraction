"""Unit tests for the Langfuse tracing wrapper."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from src.integrations.langfuse import TraceHandle, _build_usage_details


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
    def start_as_current_generation(self, **kwargs):
        self.calls.append(kwargs)
        yield self.generation


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
