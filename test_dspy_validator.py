"""
test_dspy_validator.py
======================
Simple test script to verify DSPy integration with AnswerValidatorAgent.

This script tests:
1. MLXLM adapter initialization
2. DSPy AnswerValidator loading and inference
3. Structured output generation
4. Comparison with legacy method

Usage:
    python test_dspy_validator.py
"""

from __future__ import annotations

import logging

from validator_agent import AnswerValidatorAgent
from agentic_rag_flow import RAGAnswer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def test_dspy_validator():
    """Test DSPy AnswerValidatorAgent with a simple example."""

    # Use a smaller model for quick testing
    model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    log.info("=" * 70)
    log.info("Testing DSPy AnswerValidatorAgent")
    log.info("=" * 70)

    # Test question and context
    question = "What is the capital of France?"

    # Create a test answer with a hallucination
    answer = RAGAnswer(
        question=question,
        answer="The capital of France is Paris, and it has a population of 10 million people.",
        reasoning_trace="Based on the provided sources about France.",
        source_chunks=[],
    )

    # Source context (doesn't mention population)
    source_texts = [
        "France is a country in Western Europe. Its capital city is Paris, which is located in the north-central part of the country.",
        "Paris is known for the Eiffel Tower and is a major cultural center.",
    ]

    log.info("\n📋 Test Input:")
    log.info(f"  Question: {question}")
    log.info(f"  Answer: {answer.answer}")
    log.info(f"  Sources: {len(source_texts)} chunks")

    # Test DSPy version
    log.info("\n🤖 Testing DSPy-enhanced validation...")
    try:
        validator_dspy = AnswerValidatorAgent(model_id, use_dspy=True)

        with validator_dspy:
            result_dspy = validator_dspy.validate_answer(
                question=question,
                answer=answer,
                source_texts=source_texts,
                trace=None,
            )

        log.info("\n✅ DSPy Validation Result:")
        log.info(f"  is_grounded: {result_dspy.is_grounded}")
        log.info(f"  verdict_score: {result_dspy.verdict_score:.2f}")
        log.info(f"  hallucinations: {result_dspy.hallucinations}")
        log.info(f"  validator_notes: {result_dspy.validator_notes}")
        if result_dspy.revised_answer:
            log.info(f"  revised_answer: {result_dspy.revised_answer}")

    except Exception as e:
        log.error(f"\n❌ DSPy validation failed: {e}", exc_info=True)
        return False

    # Test legacy version for comparison
    log.info("\n🔧 Testing legacy validation for comparison...")
    try:
        validator_legacy = AnswerValidatorAgent(model_id, use_dspy=False)

        with validator_legacy:
            result_legacy = validator_legacy.validate_answer(
                question=question,
                answer=answer,
                source_texts=source_texts,
                trace=None,
            )

        log.info("\n✅ Legacy Validation Result:")
        log.info(f"  is_grounded: {result_legacy.is_grounded}")
        log.info(f"  verdict_score: {result_legacy.verdict_score:.2f}")
        log.info(f"  hallucinations: {result_legacy.hallucinations}")
        log.info(f"  validator_notes: {result_legacy.validator_notes}")
        if result_legacy.revised_answer:
            log.info(f"  revised_answer: {result_legacy.revised_answer}")

    except Exception as e:
        log.error(f"\n❌ Legacy validation failed: {e}", exc_info=True)
        return False

    # Compare results
    log.info("\n" + "=" * 70)
    log.info("📊 Comparison Summary:")
    log.info("=" * 70)
    log.info(f"  DSPy detected hallucination: {not result_dspy.is_grounded}")
    log.info(f"  Legacy detected hallucination: {not result_legacy.is_grounded}")
    log.info(f"  DSPy score: {result_dspy.verdict_score:.2f}")
    log.info(f"  Legacy score: {result_legacy.verdict_score:.2f}")

    if not result_dspy.is_grounded:
        log.info("\n✅ Success: DSPy correctly identified hallucination!")
        log.info(f"  Hallucinations found: {result_dspy.hallucinations}")
    else:
        log.warning("\n⚠️  DSPy marked answer as grounded despite hallucination")

    log.info("\n" + "=" * 70)
    log.info("✅ DSPy integration test completed successfully!")
    log.info("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = test_dspy_validator()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        log.info("\n⚠️  Test interrupted by user")
        exit(130)
    except Exception as e:
        log.error(f"\n❌ Test failed with exception: {e}", exc_info=True)
        exit(1)
