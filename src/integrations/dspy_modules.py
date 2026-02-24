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


class ChunkQualityOutput(BaseModel):
    """
    Pydantic model for DSPy-based chunk validation output.

    Validates extraction quality against original source content.
    Detects fabrications, omissions, and extraction errors.

    Attributes:
        is_valid: Whether extraction is faithful and complete
        issues: Specific problems found
        corrected_structured_text: Corrected text (if needed)
        corrected_intuition_summary: Corrected summary (if needed)
        corrected_key_concepts: Corrected concepts (if needed)
        verdict_score: Extraction quality score 0-1
        validator_notes: Reasoning about validation issues
    """

    is_valid: bool = Field(description="Whether the extracted chunk faithfully represents the original content")
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found (fabrication, omissions, incorrect metadata, etc.)",
    )
    corrected_structured_text: str | None = Field(
        default=None,
        description="Corrected version of structured_text (null if no correction needed)",
    )
    corrected_intuition_summary: str | None = Field(
        default=None,
        description="Corrected version of intuition_summary (null if no correction needed)",
    )
    corrected_key_concepts: list[str] | None = Field(
        default=None,
        description="Corrected list of key concepts (null if no correction needed)",
    )
    verdict_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction quality score from 0.0 (invalid) to 1.0 (perfect)",
    )
    validator_notes: str = Field(
        default="",
        description="Brief reasoning about validation issues",
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


class ChunkQualitySignature(dspy.Signature):
    """
    DSPy signature for chunk extraction quality validation.

    Audits whether the extracted chunk faithfully and completely represents
    the original source content. Checks for fabrications, omissions, incorrect
    metadata, and other extraction errors.

    Input fields:
        original_content: The original raw content from the PDF
        extracted_text: The structured_text field extracted by the agent
        intuition_summary: The one-sentence summary provided by the agent
        key_concepts: The list of key concepts identified by the agent
        chunk_type: Type of chunk (TEXT, TABLE, or FIGURE)

    Output fields:
        is_valid: True if extraction is faithful and complete
        issues: List of specific problems found
        corrected_structured_text: Corrected text if needed
        corrected_intuition_summary: Corrected summary if needed
        corrected_key_concepts: Corrected concepts list if needed
        verdict_score: Extraction quality score between 0.0 and 1.0
        validator_notes: Brief reasoning about validation decision
    """

    original_content: str = dspy.InputField(description="The original raw content from the PDF")
    extracted_text: str = dspy.InputField(description="The structured_text field extracted by the agent")
    intuition_summary: str = dspy.InputField(description="The one-sentence intuition_summary provided by the agent")
    key_concepts: list[str] = dspy.InputField(description="The list of key_concepts identified by the agent")
    chunk_type: str = dspy.InputField(description="Type of chunk: TEXT, TABLE, or FIGURE")

    # Output fields
    is_valid: bool = dspy.OutputField(description="True if extraction is faithful and complete, False otherwise")
    issues: list[str] = dspy.OutputField(description="Specific problems found (empty list if valid)")
    corrected_structured_text: str = dspy.OutputField(
        description="Corrected structured_text (set to 'null' if no correction needed)"
    )
    corrected_intuition_summary: str = dspy.OutputField(
        description="Corrected intuition_summary (set to 'null' if no correction needed)"
    )
    corrected_key_concepts: list[str] = dspy.OutputField(
        description="Corrected key_concepts (set to 'null' if no correction needed)"
    )
    verdict_score: float = dspy.OutputField(description="Extraction quality score between 0.0 and 1.0")
    validator_notes: str = dspy.OutputField(description="Brief reasoning about validation decision")
