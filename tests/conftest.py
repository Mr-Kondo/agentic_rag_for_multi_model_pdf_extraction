"""
conftest.py - Pytest configuration and shared fixtures.

Provides reusable fixtures for test suite including:
- Temporary directories
- Mock data structures
- Test model configurations
"""

import logging
from pathlib import Path
from typing import Generator

import pytest

from src.core.models import ChunkType, ProcessedChunk, RAGAnswer, RawChunk

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@pytest.fixture
def sample_question() -> str:
    """Sample question for testing."""
    return "What are the main findings of the study?"


@pytest.fixture
def sample_raw_chunk() -> RawChunk:
    """Sample raw chunk for testing."""
    return RawChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file="test.pdf",
        raw_content="This is a sample text chunk from a PDF document.",
    )


@pytest.fixture
def sample_processed_chunk() -> ProcessedChunk:
    """Sample processed chunk for testing."""
    return ProcessedChunk(
        chunk_type=ChunkType.TEXT,
        page_num=1,
        source_file="test.pdf",
        structured_text="This is a sample text chunk from a PDF document.",
        intuition_summary="Sample text about testing",
        key_concepts=["testing", "sample", "chunk"],
        confidence=0.9,
        agent_notes="Extracted successfully",
    )


@pytest.fixture
def sample_rag_answer() -> RAGAnswer:
    """Sample RAG answer for testing."""
    return RAGAnswer(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        reasoning_trace="Based on the provided context about France.",
        source_chunks=[
            {
                "text": "France is a country in Western Europe. Paris is its capital.",
                "page": 1,
                "chunk_type": "text",
            }
        ],
    )


@pytest.fixture
def sample_source_texts() -> list[str]:
    """Sample source texts for validation testing."""
    return [
        "France is a country in Western Europe. Its capital city is Paris.",
        "Paris is known for the Eiffel Tower and is a major cultural center.",
        "The population of Paris is approximately 2.2 million in the city proper.",
    ]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary storage directory for tests."""
    storage_dir = tmp_path / "chroma_db"
    storage_dir.mkdir()
    yield storage_dir
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def test_model_config() -> dict[str, str]:
    """Test model configuration using small/fast models."""
    return {
        "text_model": "qwen3:8b",
        "table_model": "qwen2.5:3b",
        "vision_model": "qwen2.5vl:7b",
        "orchestrator_model": "deepseek-r1:8b",
        "chunk_validator_model": "qwen2.5vl:7b",
        "answer_validator_model": "qwen2.5:7b",
    }
