"""
DSPy signatures and Pydantic models for structured validation.

Defines input/output schemas for DSPy-enhanced validation agents with
type-safe Pydantic models and DSPy signature specifications.
"""

import dspy
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS (structured outputs)
# ═══════════════════════════════════════════════════════════


class AnswerGroundingOutput(BaseModel):
    """
    Pydantic model for DSPy-based answer validation output.

    Ensures structured, type-safe responses from validation LLM.
    Used to detect hallucinations and assess grounding quality.

    Attributes:
        is_grounded: Whether all claims are supported by context
        hallucinations: List of specific unsupported claims
        revised_answer: Corrected answer (if needed)
        verdict_score: Grounding quality score 0-1
        validator_notes: Reasoning about validation decision
    """

    is_grounded: bool = Field(description="Whether all material claims in the answer are supported by the source context")
    hallucinations: list[str] = Field(
        default_factory=list,
        description="List of specific unsupported claims found in the answer",
    )
    revised_answer: str | None = Field(
        default=None,
        description="Corrected answer with hallucinations removed (null if answer is already grounded)",
    )
    verdict_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Grounding quality score from 0.0 (completely ungrounded) to 1.0 (fully grounded)",
    )
    validator_notes: str = Field(
        default="",
        description="Brief reasoning about the validation decision",
    )


# ═══════════════════════════════════════════════════════════
# DSPY SIGNATURES (input/output schemas for LLM tasks)
# ═══════════════════════════════════════════════════════════


class AnswerGroundingSignature(dspy.Signature):
    """
    DSPy signature for hallucination detection and answer grounding validation.

    Verifies that every material factual claim in the answer can be traced back
    to explicit statements in the source context. Identifies hallucinations and
    provides corrected versions when necessary.

    Input fields:
        answer: The answer text to validate for hallucinations
        context: The source context text that should support all claims

    Output fields:
        is_grounded: True if all material claims are supported, False otherwise
        hallucinations: List of specific unsupported claims
        revised_answer: Corrected answer with hallucinations removed
        verdict_score: Grounding quality score between 0.0 and 1.0
        validator_notes: Brief explanation of validation decision
    """

    answer: str = dspy.InputField(description="The answer text to validate for hallucinations")
    context: str = dspy.InputField(description="The source context text that should support all claims in the answer")

    # Output fields - DSPy will structure the LLM response to match these
    is_grounded: bool = dspy.OutputField(description="True if all material claims are supported by context, False otherwise")
    hallucinations: list[str] = dspy.OutputField(
        description="List of specific unsupported claims (empty list if fully grounded)"
    )
    revised_answer: str = dspy.OutputField(
        description="Corrected answer with hallucinations removed (set to 'null' if answer is already grounded)"
    )
    verdict_score: float = dspy.OutputField(description="Grounding quality score between 0.0 and 1.0")
    validator_notes: str = dspy.OutputField(description="Brief explanation of validation decision")
