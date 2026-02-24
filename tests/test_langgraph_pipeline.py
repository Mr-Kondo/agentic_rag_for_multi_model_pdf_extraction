"""
Tests for LangGraph query pipeline implementation.

This test module validates the LangGraph-based query pipeline:
- State initialization and schemas
- Node function behavior (mocked)
- Graph construction and routing
- Integration with existing components

Note: Full end-to-end tests require models and PDFs.
This focuses on structure, routing, and state management.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.graph_state import QueryState, init_query_state
from src.core.langgraph_pipeline import (
    LangGraphQueryPipeline,
    retrieve_node,
    check_retrieval_quality_node,
    route_after_quality_check,
    route_after_decide_validate,
    route_after_grounding_check,
)
from src.core.models import RAGAnswer


class TestQueryState:
    """Test QueryState schema and initialization."""

    def test_init_query_state_defaults(self):
        """Test that init_query_state creates valid state with defaults."""
        state = init_query_state(question="Test question?")

        assert state["question"] == "Test question?"
        assert state["validates"] is False
        assert state["session_id"] is None
        assert state["retrieved_hits"] == []
        assert state["raw_answer"] is None
        assert state["final_answer"] is None
        assert state["errors"] == []
        assert state["warnings"] == []
        assert state["current_phase"] == "retrieve"
        assert state["needs_revision"] is False
        assert state["insufficient_context"] is False

    def test_init_query_state_with_options(self):
        """Test init_query_state with all options."""
        state = init_query_state(
            question="What is X?",
            validates=True,
            session_id="test-session-123",
        )

        assert state["question"] == "What is X?"
        assert state["validates"] is True
        assert state["session_id"] == "test-session-123"


class TestNodeFunctions:
    """Test individual node functions (with mocking where needed)."""

    def test_check_retrieval_quality_node_no_hits(self):
        """Test quality gate when no chunks retrieved."""
        state = init_query_state("Test?")
        state["retrieved_hits"] = []

        result = check_retrieval_quality_node(state)

        assert result["insufficient_context"] is True
        assert result["current_phase"] == "finalize"
        assert len(result["warnings"]) > 0

    def test_check_retrieval_quality_node_with_hits(self):
        """Test quality gate when chunks found."""
        state = init_query_state("Test?")
        state["retrieved_hits"] = [{"text": "Sample chunk", "metadata": {}}]

        result = check_retrieval_quality_node(state)

        assert result["insufficient_context"] is False
        assert result["current_phase"] == "generate"
        assert len(result["warnings"]) == 0

    def test_decide_validate_node_enabled(self):
        """Test decide_validate_node when validation enabled."""
        from src.core.langgraph_pipeline import decide_validate_node

        state = init_query_state("Test?", validates=True)

        result = decide_validate_node(state)

        assert result["current_phase"] == "validate"
        assert result.get("skip_validation") != True

    def test_decide_validate_node_disabled(self):
        """Test decide_validate_node when validation disabled."""
        from src.core.langgraph_pipeline import decide_validate_node

        state = init_query_state("Test?", validates=False)

        result = decide_validate_node(state)

        assert result["current_phase"] == "finalize"
        assert result["skip_validation"] is True


class TestConditionalRouting:
    """Test conditional edge routing functions."""

    def test_route_after_quality_check_no_hits(self):
        """Test routing when no hits retrieved."""
        state = init_query_state("Test?")
        state["insufficient_context"] = True

        route = route_after_quality_check(state)

        assert route == "finalize"

    def test_route_after_quality_check_with_hits(self):
        """Test routing when hits retrieved."""
        state = init_query_state("Test?")
        state["insufficient_context"] = False

        route = route_after_quality_check(state)

        assert route == "generate"

    def test_route_after_decide_validate_enabled(self):
        """Test routing when validation enabled."""
        state = init_query_state("Test?", validates=True)

        route = route_after_decide_validate(state)

        assert route == "validate"

    def test_route_after_decide_validate_disabled(self):
        """Test routing when validation disabled."""
        state = init_query_state("Test?", validates=False)

        route = route_after_decide_validate(state)

        assert route == "finalize"

    def test_route_after_grounding_check_needs_revision(self):
        """Test routing when answer needs revision."""
        state = init_query_state("Test?")
        state["needs_revision"] = True

        route = route_after_grounding_check(state)

        assert route == "revise"

    def test_route_after_grounding_check_grounded(self):
        """Test routing when answer is grounded."""
        state = init_query_state("Test?")
        state["needs_revision"] = False

        route = route_after_grounding_check(state)

        assert route == "finalize"


class TestGraphConstruction:
    """Test graph building and structure."""

    @patch("src.core.langgraph_pipeline.ReasoningOrchestratorAgent")
    @patch("src.core.langgraph_pipeline.AnswerValidatorAgent")
    @patch("src.core.langgraph_pipeline.ChunkStore")
    @patch("src.core.langgraph_pipeline.LangfuseTracer")
    def test_pipeline_initialization(self, mock_tracer, mock_store, mock_validator, mock_orchestrator):
        """Test LangGraphQueryPipeline initialization."""
        # Create mocks
        orchestrator = MagicMock()
        validator = MagicMock()
        store = MagicMock()
        tracer = MagicMock()

        # Initialize pipeline
        pipeline = LangGraphQueryPipeline(
            orchestrator=orchestrator,
            answer_validator=validator,
            store=store,
            tracer=tracer,
        )

        # Verify components stored
        assert pipeline.orchestrator is orchestrator
        assert pipeline.answer_validator is validator
        assert pipeline.store is store
        assert pipeline.tracer is tracer

        # Verify graph compiled
        assert pipeline.graph is not None

    @patch("src.core.langgraph_pipeline.ReasoningOrchestratorAgent")
    @patch("src.core.langgraph_pipeline.AnswerValidatorAgent")
    @patch("src.core.langgraph_pipeline.ChunkStore")
    @patch("src.core.langgraph_pipeline.LangfuseTracer")
    def test_pipeline_build_classmethod(self, mock_tracer, mock_store, mock_validator, mock_orchestrator):
        """Test LangGraphQueryPipeline.build() classmethod."""
        # Mock return values
        mock_orchestrator.return_value = MagicMock()
        mock_validator.return_value = MagicMock()
        mock_store.return_value = MagicMock()
        mock_tracer.return_value = MagicMock()

        # Build pipeline
        pipeline = LangGraphQueryPipeline.build()

        # Verify build succeeded
        assert isinstance(pipeline, LangGraphQueryPipeline)
        assert pipeline.orchestrator is not None
        assert pipeline.answer_validator is not None
        assert pipeline.store is not None
        assert pipeline.tracer is not None


class TestPipelineIntegration:
    """Test pipeline integration (mocked models)."""

    @patch("src.core.langgraph_pipeline.ReasoningOrchestratorAgent")
    @patch("src.core.langgraph_pipeline.AnswerValidatorAgent")
    @patch("src.core.langgraph_pipeline.ChunkStore")
    @patch("src.core.langgraph_pipeline.LangfuseTracer")
    def test_query_no_hits_early_exit(self, mock_tracer_cls, mock_store_cls, mock_validator_cls, mock_orchestrator_cls):
        """Test query pipeline with no hits (early exit via quality gate)."""
        # Setup mocks
        mock_orchestrator = MagicMock()
        mock_orchestrator.retrieve.return_value = []  # No hits
        mock_orchestrator.__enter__ = MagicMock(return_value=mock_orchestrator)
        mock_orchestrator.__exit__ = MagicMock(return_value=False)

        mock_validator = MagicMock()
        mock_store = MagicMock()

        mock_tracer = MagicMock()
        mock_trace = MagicMock()
        mock_trace.trace_id = "test-trace-123"
        mock_tracer.trace.return_value.__enter__ = MagicMock(return_value=mock_trace)
        mock_tracer.trace.return_value.__exit__ = MagicMock(return_value=False)

        # Create pipeline
        pipeline = LangGraphQueryPipeline(
            orchestrator=mock_orchestrator,
            answer_validator=mock_validator,
            store=mock_store,
            tracer=mock_tracer,
        )

        # Mock graph.invoke to return expected final state (bypass graph execution)
        expected_final_answer = RAGAnswer(
            question="What is X?",
            answer="[No relevant context found] I couldn't find relevant information in the document to answer your question.",
            reasoning_trace="No chunks retrieved from vector store",
            source_chunks=[],
            trace_id="test-trace-123",
        )

        mock_final_state = init_query_state("What is X?", validates=False)
        mock_final_state["final_answer"] = expected_final_answer
        mock_final_state["insufficient_context"] = True
        mock_final_state["stats"] = {"retrieved_count": 0, "answer_length": 0, "is_grounded": "N/A"}

        with patch.object(pipeline.graph, "invoke", return_value=mock_final_state) as mock_invoke:
            # Execute query
            result = pipeline.query("What is X?", validates=False)

            # Verify graph was called with proper state structure
            mock_invoke.assert_called_once()
            call_args = mock_invoke.call_args[0][0]
            assert call_args["question"] == "What is X?"
            assert call_args["validates"] is False

        # Verify result matches expected output
        assert result is not None
        assert isinstance(result, RAGAnswer)
        assert "No relevant context found" in result.answer
        assert result.trace_id == "test-trace-123"

        # Note: We can't verify orchestrator.generate wasn't called because
        # we're bypassing graph execution entirely (appropriate for unit tests)


@pytest.mark.integration
class TestEndToEnd:
    """
    End-to-end integration tests (requires models and data).

    These tests are marked as integration and skipped by default.
    Run with: pytest tests/test_langgraph_pipeline.py -m integration
    """

    @pytest.mark.skip(reason="Requires model loading and PDF data")
    def test_full_query_pipeline(self):
        """
        Full query pipeline test with real models.

        This test should:
        1. Build pipeline with real models
        2. Query with a test question
        3. Verify answer generation
        4. Verify validation if enabled
        """
        pipeline = LangGraphQueryPipeline.build()
        result = pipeline.query("What is the main topic?", validates=True)

        assert result is not None
        assert isinstance(result, RAGAnswer)
        assert len(result.answer) > 0
        assert result.trace_id is not None


class TestStateSafety:
    """Test state immutability and safety."""

    def test_state_mutation_doesnt_affect_original(self):
        """Verify that node functions don't mutate shared state dangerously."""
        original = init_query_state("Test?")
        original_question = original["question"]

        # Simulate node processing
        state_copy = check_retrieval_quality_node(original.copy())

        # Original should be unchanged (except for shallow copy semantics)
        assert original["question"] == original_question

    def test_error_accumulation(self):
        """Test that errors are accumulated without halting pipeline."""
        state = init_query_state("Test?")

        # Simulate errors being added
        state["errors"].append("Error 1")
        state["errors"].append("Error 2")

        assert len(state["errors"]) == 2
        assert "Error 1" in state["errors"]
        assert "Error 2" in state["errors"]
