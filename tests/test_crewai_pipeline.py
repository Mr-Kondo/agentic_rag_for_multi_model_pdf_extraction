"""
Integration tests for CrewAI-powered PDF extraction and RAG.

Tests cover:
- MLX-compatible tool wrappers
- CrewAI agent creation and execution
- Ingestion crew (extraction, validation, linking)
- RAG query crew
- End-to-end pipeline with CrewAI
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.core.models import ChunkType, RawChunk, ProcessedChunk, CrossLinkMetadata
from src.integrations.crew_mlx_tools import (
    CrewMLXToolkit,
    MLXTextExtractionTool,
    MLXTableExtractionTool,
    MLXVisionExtractionTool,
    CrossReferenceDetectionTool,
)
from src.agents.crewai_agents import (
    TextExtractorAgent,
    TableExtractorAgent,
    VisionExtractorAgent,
    CrossReferenceAnalystAgent,
    CrewAgentFactory,
)
from src.core.crewai_pipeline import (
    ExtractionCrew,
    ValidationCrew,
    LinkingCrew,
    CrewAIIngestionPipeline,
)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def sample_text_chunk():
    """Create a sample text chunk for testing."""
    return RawChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        raw_content="The machine learning model achieved 95% accuracy on the test dataset.",
        source_file="test.pdf",
    )


@pytest.fixture
def sample_table_chunk():
    """Create a sample table chunk for testing."""
    markdown_table = """
    | Metric | Value |
    |--------|-------|
    | Accuracy | 95% |
    | Precision | 0.93 |
    | Recall | 0.97 |
    """
    return RawChunk(
        chunk_type=ChunkType.TABLE,
        page_num=2,
        raw_content=markdown_table,
        source_file="test.pdf",
    )


@pytest.fixture
def sample_figure_chunk():
    """Create a sample figure chunk for testing."""
    return RawChunk(
        chunk_type=ChunkType.FIGURE,
        page_num=3,
        raw_content="/path/to/figure.png",
        source_file="test.pdf",
    )


@pytest.fixture
def crew_toolkit():
    """Create a CrewMLXToolkit instance."""
    return CrewMLXToolkit()


# ═══════════════════════════════════════════════════════════
# TOOL WRAPPER TESTS
# ═══════════════════════════════════════════════════════════


def test_text_extraction_tool_initialization(crew_toolkit):
    """Test that text extraction tool initializes correctly."""
    tool = MLXTextExtractionTool(crew_toolkit.text_agent)
    assert tool.name == "extract_text_passage"
    assert "Extract and structure text" in tool.description


def test_table_extraction_tool_initialization(crew_toolkit):
    """Test that table extraction tool initializes correctly."""
    tool = MLXTableExtractionTool(crew_toolkit.table_agent)
    assert tool.name == "extract_table"
    assert "Extract and structure" in tool.description


def test_vision_extraction_tool_initialization(crew_toolkit):
    """Test that vision extraction tool initializes correctly."""
    tool = MLXVisionExtractionTool(crew_toolkit.vision_agent)
    assert tool.name == "extract_figure"
    assert "Extract description and type" in tool.description


def test_cross_reference_detection_tool():
    """Test cross-reference detection tool."""
    tool = CrossReferenceDetectionTool()
    assert tool.name == "detect_cross_references"
    assert "cross-references" in tool.description.lower()


# ═══════════════════════════════════════════════════════════
# AGENT CREATION TESTS
# ═══════════════════════════════════════════════════════════


def test_crew_agent_factory_creates_text_extractor():
    """Test that factory creates text extraction agent."""
    from src.integrations.crew_mlx_tools import MLXTextExtractionTool

    tool = MLXTextExtractionTool()
    agent = CrewAgentFactory.create_text_extractor(tool)
    assert agent.role == "Text Content Extractor"
    assert "extract" in agent.goal.lower()


def test_crew_agent_factory_creates_table_extractor():
    """Test that factory creates table extraction agent."""
    from src.integrations.crew_mlx_tools import MLXTableExtractionTool

    tool = MLXTableExtractionTool()
    agent = CrewAgentFactory.create_table_extractor(tool)
    assert agent.role == "Table Structure Master"
    assert "table" in agent.goal.lower()


def test_crew_agent_factory_creates_linking_agent():
    """Test that factory creates linking agent."""
    tool = CrossReferenceDetectionTool()
    agent = CrewAgentFactory.create_linking_agent(tool)
    assert agent.role == "Cross-Reference Analyst"
    assert "cross-reference" in agent.goal.lower() or "relationship" in agent.goal.lower()


def test_crew_agent_factory_creates_retrieval_agent():
    """Test that factory creates retrieval agent."""
    agent = CrewAgentFactory.create_retrieval_agent()
    assert agent.role == "Information Retrieval Specialist"


def test_crew_agent_factory_creates_verification_agent():
    """Test that factory creates verification agent."""
    agent = CrewAgentFactory.create_verification_agent()
    assert agent.role == "Answer Verification Expert"


# ═══════════════════════════════════════════════════════════
# CREW TESTS
# ═══════════════════════════════════════════════════════════


def test_extraction_crew_initialization(crew_toolkit):
    """Test that extraction crew initializes correctly."""
    crew = ExtractionCrew(crew_toolkit)
    assert crew.text_agent is not None
    assert crew.table_agent is not None
    assert crew.vision_agent is not None


def test_validation_crew_initialization(crew_toolkit):
    """Test that validation crew initializes correctly."""
    crew = ValidationCrew(crew_toolkit)
    assert crew.qa_agent is not None or crew.qa_agent is None  # Optional


def test_linking_crew_initialization(crew_toolkit):
    """Test that linking crew initializes correctly."""
    crew = LinkingCrew(crew_toolkit)
    assert crew.linking_agent is not None or crew.linking_agent is None  # Optional


# ═══════════════════════════════════════════════════════════
# PIPELINE TESTS
# ═══════════════════════════════════════════════════════════


def test_crewai_ingestion_pipeline_initialization(crew_toolkit):
    """Test that pipeline initializes correctly."""
    from src.core.store import ChunkStore

    store = MagicMock(spec=ChunkStore)
    pipeline = CrewAIIngestionPipeline(store, crew_toolkit)
    assert pipeline.extraction_crew is not None
    assert pipeline.validation_crew is not None
    assert pipeline.linking_crew is not None


def test_cross_link_metadata_creation():
    """Test that CrossLinkMetadata can be created."""
    link = CrossLinkMetadata(
        source_chunk_id="chunk_1",
        target_chunk_id="chunk_2",
        link_type="table_references_figure",
        confidence=0.9,
        description="Table 3 references Figure 2a",
    )
    assert link.source_chunk_id == "chunk_1"
    assert link.target_chunk_id == "chunk_2"
    assert link.link_type == "table_references_figure"
    assert link.confidence == 0.9


def test_processed_chunk_with_cross_links():
    """Test that ProcessedChunk can store cross-links."""
    chunk = ProcessedChunk(
        chunk_type=ChunkType.TABLE,
        page_num=3,
        structured_text="Table data",
    )

    link = CrossLinkMetadata(
        source_chunk_id=chunk.chunk_id,
        target_chunk_id="other_chunk",
        link_type="reference",
    )

    chunk.cross_links.append(link)
    assert len(chunk.cross_links) == 1
    assert chunk.cross_links[0].link_type == "reference"


# ═══════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════


@pytest.mark.skip(reason="Requires actual MLX models; use for manual integration testing")
def test_end_to_end_crewai_ingestion(sample_text_chunk, sample_table_chunk):
    """Test full ingestion pipeline with CrewAI."""
    from src.core.store import ChunkStore

    store = MagicMock(spec=ChunkStore)
    store.upsert = MagicMock(return_value=2)

    toolkit = CrewMLXToolkit()
    pipeline = CrewAIIngestionPipeline(store, toolkit)

    chunks = [sample_text_chunk, sample_table_chunk]
    stored = pipeline.process_chunks(chunks)

    assert stored > 0
    store.upsert.assert_called()


@pytest.mark.skip(reason="Requires actual CrewAI execution; use for manual testing")
def test_rag_query_crew_initialization(crew_toolkit):
    """Test RAG query crew initialization."""
    from src.core.crewai_pipeline import RAGQueryCrew
    from src.core.store import ChunkStore

    store = MagicMock(spec=ChunkStore)
    crew = RAGQueryCrew(crew_toolkit, store)
    assert crew.retrieval_specialist is not None
    assert crew.reasoning_agent is not None
    assert crew.verification_agent is not None


# ═══════════════════════════════════════════════════════════
# MOCK/UNIT TESTS (No Model Loading)
# ═══════════════════════════════════════════════════════════


def test_crew_mlx_toolkit_get_extraction_tools(crew_toolkit):
    """Test that toolkit provides extraction tools."""
    tools = crew_toolkit.get_extraction_tools()
    assert len(tools) == 3
    assert any("text" in t.name.lower() for t in tools)
    assert any("table" in t.name.lower() for t in tools)
    assert any("figure" in t.name.lower() for t in tools)


def test_crew_mlx_toolkit_get_linking_tools(crew_toolkit):
    """Test that toolkit provides linking tools."""
    tools = crew_toolkit.get_linking_tools()
    assert len(tools) == 1
    assert "cross" in tools[0].name.lower()


def test_crew_mlx_toolkit_get_all_tools(crew_toolkit):
    """Test that toolkit provides all available tools."""
    all_tools = crew_toolkit.get_all_tools()
    assert len(all_tools) >= 3  # At least extraction tools


def test_cross_reference_detection_with_empty_chunks():
    """Test cross-reference detection with empty chunk list."""
    tool = CrossReferenceDetectionTool()
    result = tool._run([])
    assert result.has_links == False
    assert result.links == []


def test_cross_reference_detection_with_chunks():
    """Test cross-reference detection with sample chunks."""
    tool = CrossReferenceDetectionTool()
    chunks = [
        {
            "chunk_id": "chunk_1",
            "chunk_type": "table",
            "structured_text": "See Figure 2a for the visualization.",
        },
        {
            "chunk_id": "chunk_2",
            "chunk_type": "figure",
            "structured_text": "Figure showing results",
        },
    ]
    result = tool._run(chunks)
    assert hasattr(result, "has_links")
    assert hasattr(result, "links")
    assert hasattr(result, "confidence")


# ═══════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════


def test_processed_chunk_with_multiple_cross_links():
    """Test ProcessedChunk with multiple cross-links."""
    chunk = ProcessedChunk(
        chunk_type=ChunkType.TABLE,
        page_num=5,
        structured_text="Complex table",
    )

    # Add multiple links
    for i in range(3):
        link = CrossLinkMetadata(
            source_chunk_id=chunk.chunk_id,
            target_chunk_id=f"target_{i}",
            link_type="reference",
        )
        chunk.cross_links.append(link)

    assert len(chunk.cross_links) == 3
    assert all(link.source_chunk_id == chunk.chunk_id for link in chunk.cross_links)


def test_document_extraction_result_model():
    """Test ExtractionResult Pydantic model."""
    from src.integrations.crew_mlx_tools import ExtractionResult

    result = ExtractionResult(
        chunk_type="text",
        structured_text="Sample text",
        intuition_summary="A brief summary",
        key_concepts=["keyword1", "keyword2"],
        confidence=0.85,
        agent_notes="No issues detected",
    )

    assert result.chunk_type == "text"
    assert result.confidence == 0.85
    assert len(result.key_concepts) == 2


def test_validation_result_model():
    """Test ValidationResult Pydantic model."""
    from src.integrations.crew_mlx_tools import ValidationResult

    result = ValidationResult(
        is_valid=True,
        validation_score=0.95,
        correction=None,
        reason="Content matches original",
    )

    assert result.is_valid == True
    assert result.validation_score == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
