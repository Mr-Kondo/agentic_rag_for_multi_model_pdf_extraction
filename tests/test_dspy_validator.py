"""Unit tests for AnswerValidatorAgent behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agents.validation import AnswerValidatorAgent
from src.core.models import RAGAnswer


def _answer(text: str) -> RAGAnswer:
    return RAGAnswer(
        question="What is the capital of France?",
        answer=text,
        reasoning_trace="Based on retrieved sources.",
        source_chunks=[],
    )


def test_validate_answer_uses_dspy_predictor() -> None:
    """DSPy mode should return structured values from the predictor."""
    validator = AnswerValidatorAgent("qwen2.5:7b", use_dspy=True)
    validator._loaded = True
    validator._dspy_predictor = MagicMock(
        return_value=SimpleNamespace(
            is_grounded=False,
            hallucinations=["Population claim is unsupported"],
            revised_answer="The capital of France is Paris.",
            verdict_score="0.25",
            validator_notes="Population was not supported by the sources.",
        )
    )

    result = validator.validate_answer(
        question="What is the capital of France?",
        answer=_answer("The capital of France is Paris, and it has a population of 10 million people."),
        source_texts=["France is a country in Western Europe. Its capital is Paris."],
        trace=None,
    )

    assert result.is_grounded is False
    assert result.hallucinations == ["Population claim is unsupported"]
    assert result.revised_answer == "The capital of France is Paris."
    assert result.verdict_score == 0.25
    assert "Population" in result.validator_notes


def test_validate_answer_dspy_fallback_on_predictor_error() -> None:
    """DSPy mode should degrade safely when the predictor raises an exception."""
    validator = AnswerValidatorAgent("qwen2.5:7b", use_dspy=True)
    validator._loaded = True

    def _raise(*args, **kwargs):
        raise RuntimeError("predictor exploded")

    validator._dspy_predictor = _raise

    result = validator.validate_answer(
        question="What is the capital of France?",
        answer=_answer("Paris is the capital of France."),
        source_texts=["France is a country in Europe. Paris is its capital city."],
        trace=None,
    )

    assert result.is_grounded is True
    assert result.hallucinations == []
    assert result.revised_answer is None
    assert result.verdict_score == 0.5
    assert "predictor exploded" in result.validator_notes


def test_validate_answer_legacy_strips_think_blocks() -> None:
    """Legacy mode should strip <think> blocks before parsing JSON output."""
    validator = AnswerValidatorAgent("qwen2.5:7b", use_dspy=False)
    validator._loaded = True
    validator._client = MagicMock()
    validator._client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(
            content=(
                "<think>internal reasoning</think>\n"
                '{"is_grounded": false, "hallucinations": ["Population claim"], '
                '"revised_answer": "The capital of France is Paris.", '
                '"verdict_score": 0.4, "validator_notes": "Unsupported population claim."}'
            )
        )
    )

    result = validator.validate_answer(
        question="What is the capital of France?",
        answer=_answer("The capital of France is Paris, and it has a population of 10 million people."),
        source_texts=["France is a country in Europe. Paris is its capital city."],
        trace=None,
    )

    assert result.is_grounded is False
    assert result.hallucinations == ["Population claim"]
    assert result.revised_answer == "The capital of France is Paris."
    assert result.verdict_score == 0.4


@patch("src.agents.validation.dspy.ChainOfThought")
@patch("src.agents.validation.configure_ollama_lm")
def test_do_load_configures_dspy_predictor(configure_mock, cot_mock) -> None:
    """Loading DSPy mode should configure the LM and create the predictor."""
    predictor = MagicMock()
    cot_mock.return_value = predictor

    validator = AnswerValidatorAgent("qwen2.5:7b", use_dspy=True)
    validator._do_load()

    configure_mock.assert_called_once()
    cot_mock.assert_called_once()
    assert validator._dspy_predictor is predictor


@patch("src.agents.validation._model_cache.load_text_model")
def test_do_load_legacy_initializes_client(load_text_model_mock) -> None:
    """Loading legacy mode should request an Ollama client from the model cache."""
    client = MagicMock()
    load_text_model_mock.return_value = client

    validator = AnswerValidatorAgent("qwen2.5:7b", use_dspy=False)
    validator._do_load()

    load_text_model_mock.assert_called_once_with("qwen2.5:7b")
    assert validator._client is client
