"""
MLX-compatible CrewAI tool wrappers.

Bridges existing MLX-based agents with CrewAI's tool interface.
Handles load/unload lifecycle for memory-constrained devices.
"""

import json
import logging
from typing import Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.agents.extraction import TextAgent, TableAgent, VisionAgent
from src.agents.validation import ChunkValidatorAgent, AnswerValidatorAgent
from src.agents.orchestrator import ReasoningOrchestratorAgent
from src.core.models import ProcessedChunk, RawChunk, ChunkType, RAGAnswer
from src.integrations.langfuse import get_trace

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# TOOL OUTPUT MODELS (for structured CrewAI responses)
# ═══════════════════════════════════════════════════════════


class ExtractionResult(BaseModel):
    """Result of text/table/figure extraction."""

    chunk_type: str = Field(description="Type of chunk: TEXT, TABLE, or FIGURE")
    structured_text: str = Field(description="Extracted and structured content")
    intuition_summary: str = Field(description="One-sentence summary")
    key_concepts: list[str] = Field(description="Key terms extracted")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    agent_notes: str = Field(description="Notes about extraction quality or issues")


class ValidationResult(BaseModel):
    """Result of chunk validation."""

    is_valid: bool = Field(description="Whether chunk passed validation")
    validation_score: float = Field(description="Validation score (0.0-1.0)")
    correction: Optional[str] = Field(default=None, description="Corrected content if applicable")
    reason: str = Field(description="Validation reasoning")


class CrossLinkResult(BaseModel):
    """Result of cross-reference detection."""

    has_links: bool = Field(description="Whether cross-links were detected")
    links: list[dict] = Field(description="Detected cross-references")
    confidence: float = Field(description="Confidence in link detection")


class RAGResult(BaseModel):
    """Result of RAG answer generation."""

    answer: str = Field(description="Generated answer")
    sources: list[str] = Field(description="Source chunk IDs")
    reasoning: str = Field(description="Reasoning trace")
    confidence: float = Field(description="Answer confidence")


# ═══════════════════════════════════════════════════════════
# EXTRACTION TOOLS
# ═══════════════════════════════════════════════════════════


class MLXTextExtractionTool(BaseTool):
    """
    Extract structured text from PDF passages.

    Uses small text LLM (2-4B) for cleaning and structuring raw text.
    """

    name: str = "extract_text_passage"
    description: str = (
        "Extract and structure text from a PDF passage. Returns JSON with structured_text, summary, concepts, and confidence."
    )

    def __init__(self, text_agent: Optional[TextAgent] = None, **kwargs):
        """Initialize text extraction tool."""
        super().__init__(**kwargs)
        self.text_agent = text_agent or TextAgent(model_id="mlx-community/Phi-3.5-mini-instruct")

    def _run(self, passage: str, page_num: int = 0) -> ExtractionResult:
        """
        Execute text extraction.

        Args:
            passage: Raw text to extract
            page_num: Page number for tracking

        Returns:
            ExtractionResult with structured output
        """
        try:
            trace = get_trace()
            raw_chunk = RawChunk(
                chunk_type=ChunkType.TEXT,
                page_num=page_num,
                raw_content=passage,
                source_file="<from_crew>",
            )
            processed = self.text_agent.process(raw_chunk, trace=trace)

            return ExtractionResult(
                chunk_type=processed.chunk_type.value,
                structured_text=processed.structured_text,
                intuition_summary=processed.intuition_summary,
                key_concepts=processed.key_concepts,
                confidence=processed.confidence,
                agent_notes=processed.agent_notes,
            )
        except Exception as e:
            log.error("Text extraction failed: %s", e, exc_info=True)
            raise


class MLXTableExtractionTool(BaseTool):
    """
    Extract and structure data from tables.

    Uses small LLM to enhance markdown table representations with schema.
    """

    name: str = "extract_table"
    description: str = "Extract and structure a table from PDF. Returns JSON with columns, schema, and confidence."

    def __init__(self, table_agent: Optional[TableAgent] = None, **kwargs):
        """Initialize table extraction tool."""
        super().__init__(**kwargs)
        self.table_agent = table_agent or TableAgent(model_id="mlx-community/Qwen2.5-3B-Instruct")

    def _run(self, table_markdown: str, page_num: int = 0) -> ExtractionResult:
        """
        Execute table extraction.

        Args:
            table_markdown: Markdown table string
            page_num: Page number for tracking

        Returns:
            ExtractionResult with table structure
        """
        try:
            trace = get_trace()
            raw_chunk = RawChunk(
                chunk_type=ChunkType.TABLE,
                page_num=page_num,
                raw_content=table_markdown,
                source_file="<from_crew>",
            )
            processed = self.table_agent.process(raw_chunk, trace=trace)

            return ExtractionResult(
                chunk_type=processed.chunk_type.value,
                structured_text=processed.structured_text,
                intuition_summary=processed.intuition_summary,
                key_concepts=processed.key_concepts,
                confidence=processed.confidence,
                agent_notes=processed.agent_notes,
            )
        except Exception as e:
            log.error("Table extraction failed: %s", e, exc_info=True)
            raise


class MLXVisionExtractionTool(BaseTool):
    """
    Classify and describe figures, charts, and diagrams.

    Uses vision-language model (256M) to analyze and describe visual content.
    """

    name: str = "extract_figure"
    description: str = (
        "Extract description and type classification from a figure/chart. Returns figure_type, description, and confidence."
    )

    def __init__(self, vision_agent: Optional[VisionAgent] = None, **kwargs):
        """Initialize vision extraction tool."""
        super().__init__(**kwargs)
        self.vision_agent = vision_agent or VisionAgent(model_id="mlx-community/SmolVLM-256M")

    def _run(self, image_path: str, page_num: int = 0) -> ExtractionResult:
        """
        Execute figure extraction.

        Args:
            image_path: Path to image file
            page_num: Page number for tracking

        Returns:
            ExtractionResult with figure description
        """
        try:
            trace = get_trace()
            raw_chunk = RawChunk(
                chunk_type=ChunkType.FIGURE,
                page_num=page_num,
                raw_content=image_path,
                source_file="<from_crew>",
            )
            processed = self.vision_agent.process(raw_chunk, trace=trace)

            return ExtractionResult(
                chunk_type=processed.chunk_type.value,
                structured_text=processed.structured_text,
                intuition_summary=processed.intuition_summary,
                key_concepts=processed.key_concepts,
                confidence=processed.confidence,
                agent_notes=processed.agent_notes,
            )
        except Exception as e:
            log.error("Figure extraction failed: %s", e, exc_info=True)
            raise


# ═══════════════════════════════════════════════════════════
# VALIDATION TOOLS
# ═══════════════════════════════════════════════════════════


class MLXChunkValidationTool(BaseTool):
    """
    Validate extracted chunks against original content.

    Uses vision-language model to ensure quality and accuracy.
    """

    name: str = "validate_chunk"
    description: str = (
        "Validate extracted chunk against original PDF content. Returns is_valid, validation_score, and correction if needed."
    )

    def __init__(self, validator: Optional[ChunkValidatorAgent] = None, **kwargs):
        """Initialize validation tool."""
        super().__init__(**kwargs)
        self.validator = validator

    def _run(self, structured_content: str, original_content: str) -> ValidationResult:
        """
        Validate chunk.

        Args:
            structured_content: Extracted and processed content
            original_content: Original PDF content

        Returns:
            ValidationResult with validity assessment
        """
        if self.validator is None:
            log.warning("ChunkValidatorAgent not initialized; skipping validation")
            return ValidationResult(
                is_valid=True,
                validation_score=0.8,
                reason="Validator not available; assuming valid",
            )

        try:
            trace = get_trace()
            with self.validator:
                result = self.validator.validate(
                    structured_text=structured_content,
                    original_content=original_content,
                    trace=trace,
                )
                return ValidationResult(
                    is_valid=result.get("is_valid", True),
                    validation_score=result.get("validation_score", 0.8),
                    correction=result.get("correction"),
                    reason=result.get("reason", "Validation complete"),
                )
        except Exception as e:
            log.error("Chunk validation failed: %s", e, exc_info=True)
            raise


class MLXAnswerValidationTool(BaseTool):
    """
    Detect hallucinations in RAG answers.

    Uses large reasoning model with DSPy to validate answer grounding.
    """

    name: str = "validate_answer"
    description: str = "Validate RAG answer for hallucinations and grounding in source material. Returns is_grounded, corrections, and confidence."

    def __init__(self, validator: Optional[AnswerValidatorAgent] = None, **kwargs):
        """Initialize answer validation tool."""
        super().__init__(**kwargs)
        self.validator = validator

    def _run(self, answer: str, source_chunks: list[str]) -> ValidationResult:
        """
        Validate answer grounding.

        Args:
            answer: Generated RAG answer
            source_chunks: Source chunk texts

        Returns:
            ValidationResult with grounding assessment
        """
        if self.validator is None:
            log.warning("AnswerValidatorAgent not initialized; skipping validation")
            return ValidationResult(
                is_valid=True,
                validation_score=0.8,
                reason="Validator not available; assuming valid",
            )

        try:
            trace = get_trace()
            with self.validator:
                result = self.validator.validate_answer(
                    answer=answer,
                    source_chunks=source_chunks,
                    trace=trace,
                )
                return ValidationResult(
                    is_valid=result.get("is_grounded", True),
                    validation_score=result.get("confidence", 0.8),
                    correction=result.get("revised_answer"),
                    reason=result.get("reasoning", "Validation complete"),
                )
        except Exception as e:
            log.error("Answer validation failed: %s", e, exc_info=True)
            raise


# ═══════════════════════════════════════════════════════════
# LINKING TOOL (for cross-reference detection)
# ═══════════════════════════════════════════════════════════


class CrossReferenceDetectionTool(BaseTool):
    """
    Detect cross-references between tables and figures.

    Identifies when table cells or captions reference other content.
    """

    name: str = "detect_cross_references"
    description: str = (
        "Detect cross-references between tables, figures, and text sections. Returns list of detected links with confidence."
    )

    def _run(self, chunks: list[dict]) -> CrossLinkResult:
        """
        Detect cross-references.

        Args:
            chunks: List of processed chunks with metadata

        Returns:
            CrossLinkResult with detected links
        """
        # This is a placeholder for the linking logic
        # In the full implementation, this would use a small LLM
        # to analyze relationships between chunks
        links = []

        try:
            # Parse chunks and look for references
            for chunk in chunks:
                chunk_type = chunk.get("chunk_type", "")
                content = chunk.get("structured_text", "")

                # Simple pattern matching for cross-references
                # "See Figure X", "Table Y shows", etc.
                if any(ref in content.lower() for ref in ["see figure", "table", "section", "refer to"]):
                    links.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "reference_type": "cross_reference",
                            "confidence": 0.8,
                        }
                    )

            return CrossLinkResult(
                has_links=len(links) > 0,
                links=links,
                confidence=0.85,
            )
        except Exception as e:
            log.error("Cross-reference detection failed: %s", e, exc_info=True)
            raise


# ═══════════════════════════════════════════════════════════
# RAG GENERATION TOOL
# ═══════════════════════════════════════════════════════════


class MLXRAGGenerationTool(BaseTool):
    """
    Generate answers from retrieved chunks.

    Uses large reasoning model with retrieval-augmented generation.
    """

    name: str = "generate_rag_answer"
    description: str = (
        "Generate answer from retrieved chunks using reasoning model. Returns answer, reasoning trace, and confidence."
    )

    def __init__(self, orchestrator: Optional[ReasoningOrchestratorAgent] = None, **kwargs):
        """Initialize RAG generation tool."""
        super().__init__(**kwargs)
        self.orchestrator = orchestrator

    def _run(self, question: str, retrieved_chunks: list[str]) -> RAGResult:
        """
        Generate RAG answer.

        Args:
            question: User question
            retrieved_chunks: Top-k relevant chunks

        Returns:
            RAGResult with answer and reasoning
        """
        if self.orchestrator is None:
            raise ValueError("ReasoningOrchestratorAgent not initialized")

        try:
            trace = get_trace()
            with self.orchestrator:
                result = self.orchestrator.generate(
                    question=question,
                    chunks=retrieved_chunks,
                    trace=trace,
                )

                return RAGResult(
                    answer=result.answer,
                    sources=result.metadata.get("source_chunk_ids", []) if result.metadata else [],
                    reasoning=result.metadata.get("reasoning", "") if result.metadata else "",
                    confidence=result.metadata.get("confidence", 0.8) if result.metadata else 0.8,
                )
        except Exception as e:
            log.error("RAG generation failed: %s", e, exc_info=True)
            raise


# ═══════════════════════════════════════════════════════════
# TOOL FACTORY
# ═══════════════════════════════════════════════════════════


class CrewMLXToolkit:
    """
    Factory for creating and managing MLX-compatible CrewAI tools.

    Centralizes tool initialization and lifecycle management.
    """

    def __init__(
        self,
        text_agent: Optional[TextAgent] = None,
        table_agent: Optional[TableAgent] = None,
        vision_agent: Optional[VisionAgent] = None,
        chunk_validator: Optional[ChunkValidatorAgent] = None,
        answer_validator: Optional[AnswerValidatorAgent] = None,
        orchestrator: Optional[ReasoningOrchestratorAgent] = None,
    ):
        """Initialize toolkit with agents."""
        self.text_agent = text_agent or TextAgent(model_id="mlx-community/Phi-3.5-mini-instruct")
        self.table_agent = table_agent or TableAgent(model_id="mlx-community/Qwen2.5-3B-Instruct")
        self.vision_agent = vision_agent or VisionAgent(model_id="mlx-community/SmolVLM-256M")
        self.chunk_validator = chunk_validator
        self.answer_validator = answer_validator
        self.orchestrator = orchestrator

    def get_extraction_tools(self) -> list[BaseTool]:
        """Get text/table/vision extraction tools."""
        return [
            MLXTextExtractionTool(self.text_agent),
            MLXTableExtractionTool(self.table_agent),
            MLXVisionExtractionTool(self.vision_agent),
        ]

    def get_validation_tools(self) -> list[BaseTool]:
        """Get validation tools."""
        tools = []
        if self.chunk_validator:
            tools.append(MLXChunkValidationTool(self.chunk_validator))
        if self.answer_validator:
            tools.append(MLXAnswerValidationTool(self.answer_validator))
        return tools

    def get_linking_tools(self) -> list[BaseTool]:
        """Get cross-reference detection tools."""
        return [CrossReferenceDetectionTool()]

    def get_rag_tools(self) -> list[BaseTool]:
        """Get RAG generation tools."""
        if not self.orchestrator:
            raise ValueError("ReasoningOrchestratorAgent required for RAG tools")
        return [MLXRAGGenerationTool(self.orchestrator)]

    def get_all_tools(self) -> list[BaseTool]:
        """Get all available tools."""
        tools = self.get_extraction_tools()
        tools.extend(self.get_validation_tools())
        tools.extend(self.get_linking_tools())
        try:
            tools.extend(self.get_rag_tools())
        except ValueError:
            log.debug("RAG tools not available; skipping")
        return tools
