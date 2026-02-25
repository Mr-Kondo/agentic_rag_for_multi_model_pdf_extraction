"""
CrewAI agent definitions for PDF extraction and RAG workflows.

Defines agent roles with explicit backstories and goals for multi-agent
coordination via CrewAI's task-based framework.
"""

import logging

from crewai import Agent

from src.integrations.crew_mlx_tools import (
    MLXTextExtractionTool,
    MLXTableExtractionTool,
    MLXVisionExtractionTool,
    MLXChunkValidationTool,
    CrossReferenceDetectionTool,
    MLXRAGGenerationTool,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# EXTRACTION AGENTS
# ═══════════════════════════════════════════════════════════


class TextExtractorAgent(Agent):
    """
    Agent specialized in extracting structured text from PDF passages.

    Role: Converts raw PDF text into clean, searchable passages with
    metadata (summary, concepts, confidence score).
    """

    def __init__(self, text_extraction_tool: MLXTextExtractionTool):
        """Initialize text extraction agent."""
        super().__init__(
            role="Text Content Extractor",
            goal="Extract and structure plain text passages from PDF documents into clean, properly formatted text with key concepts identified",
            backstory=(
                "You are an expert document analyst with deep experience in PDF processing. "
                "You specialize in identifying main content sections, cleaning up formatting artifacts, "
                "and extracting key concepts that will help researchers find relevant information. "
                "You work with precision and consistency, always flagging ambiguous or low-confidence extractions. "
                "Your goal is to transform raw PDF text into high-quality structured passages."
            ),
            tools=[text_extraction_tool],
            verbose=True,
            allow_delegation=False,
        )


class TableExtractorAgent(Agent):
    """
    Agent specialized in extracting structured data from tables.

    Role: Converts markdown table representations into properly formatted
    tables with schema and metadata.
    """

    def __init__(self, table_extraction_tool: MLXTableExtractionTool):
        """Initialize table extraction agent."""
        super().__init__(
            role="Table Structure Master",
            goal="Extract and enhance table structures with accurate column definitions, data types, and metadata to enable precise data analysis",
            backstory=(
                "You are a data structure specialist with expertise in table normalization and schema inference. "
                "You understand different table formats and can identify column headers, data types, units, and relationships. "
                "Your work ensures that tabular data remains queryable and analyzable even when extracted from complex PDF layouts. "
                "You take pride in data quality and consistency across all extracted tables."
            ),
            tools=[table_extraction_tool],
            verbose=True,
            allow_delegation=False,
        )


class VisionExtractorAgent(Agent):
    """
    Agent specialized in analyzing figures, charts, and visual elements.

    Role: Classifies visual content type and provides comprehensive
    descriptions for search and retrieval.
    """

    def __init__(self, vision_extraction_tool: MLXVisionExtractionTool):
        """Initialize vision extraction agent."""
        super().__init__(
            role="Visual Content Analyzer",
            goal="Analyze and describe figures, charts, and diagrams to enable accurate retrieval and understanding of visual information from PDFs",
            backstory=(
                "You are a visual content specialist trained to identify and describe complex scientific figures, "
                "charts, diagrams, and other visual elements. Your expertise spans bar charts, line graphs, "
                "scatter plots, flowcharts, maps, photographs, and network diagrams. "
                "You provide detailed descriptions including axis labels, legend values, and trends visible in the data. "
                "Your detailed descriptions help researchers understand visual content even without seeing the image."
            ),
            tools=[vision_extraction_tool],
            verbose=True,
            allow_delegation=False,
        )


# ═══════════════════════════════════════════════════════════
# VALIDATION & QUALITY AGENTS
# ═══════════════════════════════════════════════════════════


class QualityAssuranceAgent(Agent):
    """
    Agent responsible for validating extracted content quality.

    Role: Ensures extracted chunks match original PDF content and
    meet quality standards.
    """

    def __init__(self, validation_tool: MLXChunkValidationTool):
        """Initialize QA agent."""
        super().__init__(
            role="Quality Assurance Validator",
            goal="Validate that extracted content accurately represents the original PDF material with no loss of critical information or introduction of errors",
            backstory=(
                "You are a meticulous quality control expert with experience in content validation and accuracy assessment. "
                "You compare extracted content against original sources to identify discrepancies, assess information loss, "
                "and detect potential hallucinations or misinterpretations. "
                "You maintain strict quality standards and provide detailed feedback on any issues found. "
                "Your validation ensures that the RAG system provides only accurate, grounded information to users."
            ),
            tools=[validation_tool],
            verbose=True,
            allow_delegation=False,
        )


# ═══════════════════════════════════════════════════════════
# LINKING & CROSS-REFERENCE AGENT
# ═══════════════════════════════════════════════════════════


class CrossReferenceAnalystAgent(Agent):
    """
    Agent specialized in detecting relationships between content elements.

    Role: Identifies when tables, figures, and text sections reference
    each other or share information.
    """

    def __init__(self, linking_tool: CrossReferenceDetectionTool):
        """Initialize cross-reference agent."""
        super().__init__(
            role="Cross-Reference Analyst",
            goal="Identify and document relationships between tables, figures, text sections, and other content elements to enable comprehensive content understanding",
            backstory=(
                "You are a knowledge graph specialist trained to identify relationships and cross-references in technical documents. "
                "You understand how tables reference figures, how text sections cite data, and how related concepts link together. "
                "Your work creates semantic connections that help users understand the full context of information and discover related content. "
                "You maintain high accuracy in identifying meaningful relationships while avoiding false connections."
            ),
            tools=[linking_tool],
            verbose=True,
            allow_delegation=False,
        )


# ═══════════════════════════════════════════════════════════
# RAG AGENTS
# ═══════════════════════════════════════════════════════════


class RetrievalSpecialistAgent(Agent):
    """
    Agent responsible for efficiently retrieving relevant content.

    Role: Finds most relevant chunks for a user query without hallucination.
    """

    def __init__(self):
        """Initialize retrieval agent."""
        super().__init__(
            role="Information Retrieval Specialist",
            goal="Efficiently retrieve the most relevant chunks from the document database that match user queries with minimum hallucination",
            backstory=(
                "You are an expert in information retrieval with deep understanding of semantic search, "
                "ranking algorithms, and relevance assessment. "
                "You focus on precision and recall, understanding that quality of retrieval directly impacts "
                "the quality of final answers. "
                "You are conservative in what you return—better to retrieve less than to retrieve irrelevant material."
            ),
            verbose=True,
            allow_delegation=False,
        )


class ReasoningAgentMLX(Agent):
    """
    Agent that generates answers using reasoning model and retrieved context.

    Role: Synthesizes retrieved chunks into coherent, grounded answers
    using step-by-step reasoning.
    """

    def __init__(self, rag_tool: MLXRAGGenerationTool):
        """Initialize reasoning agent."""
        super().__init__(
            role="Reasoning & Synthesis Expert",
            goal="Generate comprehensive, well-reasoned answers to user questions based on retrieved document chunks with clear reasoning traces",
            backstory=(
                "You are an expert researcher with strong analytical and synthesis skills. "
                "You excel at taking multiple sources of information and weaving them into coherent, comprehensive answers. "
                "Your responses include clear reasoning traces that show how conclusions were derived from source material. "
                "You prioritize accuracy and transparency, always indicating confidence levels and limitations."
            ),
            tools=[rag_tool],
            verbose=True,
            allow_delegation=False,
        )


class AnswerVerificationAgent(Agent):
    """
    Agent responsible for verifying answer grounding and accuracy.

    Role: Detects hallucinations and ensures answers are properly grounded
    in source material.
    """

    def __init__(self):
        """Initialize answer verification agent."""
        super().__init__(
            role="Answer Verification Expert",
            goal="Verify that generated answers are accurately grounded in source material without hallucination or unsupported claims",
            backstory=(
                "You are a critical evaluator with expertise in fact-checking, logical reasoning, and hallucination detection. "
                "You meticulously examine generated answers to ensure every claim is traceable to source material. "
                "You identify unsupported inferences, exaggerations, and logical fallacies. "
                "Your feedback is constructive and specific, helping improve answer quality and user trust."
            ),
            verbose=True,
            allow_delegation=False,
        )


# ═══════════════════════════════════════════════════════════
# AGENT FACTORY
# ═══════════════════════════════════════════════════════════


class CrewAgentFactory:
    """
    Factory for creating properly configured CrewAI agents.

    Centralizes agent instantiation and ensures consistent configuration.
    """

    @staticmethod
    def create_text_extractor(text_tool: MLXTextExtractionTool) -> TextExtractorAgent:
        """Create text extraction agent."""
        return TextExtractorAgent(text_tool)

    @staticmethod
    def create_table_extractor(table_tool: MLXTableExtractionTool) -> TableExtractorAgent:
        """Create table extraction agent."""
        return TableExtractorAgent(table_tool)

    @staticmethod
    def create_vision_extractor(vision_tool: MLXVisionExtractionTool) -> VisionExtractorAgent:
        """Create vision extraction agent."""
        return VisionExtractorAgent(vision_tool)

    @staticmethod
    def create_qa_agent(validation_tool: MLXChunkValidationTool) -> QualityAssuranceAgent:
        """Create quality assurance agent."""
        return QualityAssuranceAgent(validation_tool)

    @staticmethod
    def create_linking_agent(linking_tool: CrossReferenceDetectionTool) -> CrossReferenceAnalystAgent:
        """Create cross-reference agent."""
        return CrossReferenceAnalystAgent(linking_tool)

    @staticmethod
    def create_retrieval_agent() -> RetrievalSpecialistAgent:
        """Create retrieval agent."""
        return RetrievalSpecialistAgent()

    @staticmethod
    def create_reasoning_agent(rag_tool: MLXRAGGenerationTool) -> ReasoningAgentMLX:
        """Create reasoning agent."""
        return ReasoningAgentMLX(rag_tool)

    @staticmethod
    def create_verification_agent() -> AnswerVerificationAgent:
        """Create answer verification agent."""
        return AnswerVerificationAgent()
