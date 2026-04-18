"""Tests for CLI runtime extraction mode toggles."""

import argparse

from app import _apply_runtime_pipeline_mode, create_parser


class _DummyConfig:
    """Minimal config stub for runtime override tests."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def set(self, key: str, value: object) -> None:
        self.calls.append((key, value))


def test_cli_mode_flags_are_exclusive() -> None:
    parser = create_parser()

    args = parser.parse_args(["ingest", "sample.pdf", "--quality-mode"])
    assert args.quality_mode is True
    assert args.fast_mode is False

    args = parser.parse_args(["ingest", "sample.pdf", "--fast-mode"])
    assert args.fast_mode is True
    assert args.quality_mode is False


def test_apply_runtime_pipeline_mode_sets_quality(monkeypatch) -> None:
    dummy = _DummyConfig()
    monkeypatch.setattr("app.config", dummy)

    args = argparse.Namespace(quality_mode=True, fast_mode=False)
    _apply_runtime_pipeline_mode(args)

    assert dummy.calls == [("pipeline.text_passthrough", False)]


def test_apply_runtime_pipeline_mode_sets_fast(monkeypatch) -> None:
    dummy = _DummyConfig()
    monkeypatch.setattr("app.config", dummy)

    args = argparse.Namespace(quality_mode=False, fast_mode=True)
    _apply_runtime_pipeline_mode(args)

    assert dummy.calls == [("pipeline.text_passthrough", True)]


def test_apply_runtime_pipeline_mode_no_flag_does_nothing(monkeypatch) -> None:
    dummy = _DummyConfig()
    monkeypatch.setattr("app.config", dummy)

    args = argparse.Namespace(quality_mode=False, fast_mode=False)
    _apply_runtime_pipeline_mode(args)

    assert dummy.calls == []
