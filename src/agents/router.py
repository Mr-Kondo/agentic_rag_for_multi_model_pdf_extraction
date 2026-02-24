"""
Agent router for dispatching chunks to appropriate extraction agents.

Routes based on chunk type (TEXT, TABLE, FIGURE) to specialized agents.
"""

import logging
from typing import TYPE_CHECKING

from src.core.models import ChunkType

if TYPE_CHECKING:
    from src.agents.extraction import TableAgent, TextAgent, VisionAgent
    from src.core.models import ProcessedChunk, RawChunk
    from src.integrations.langfuse import TraceHandle

log = logging.getLogger(__name__)


class AgentRouter:
    """
    Routes raw chunks to appropriate extraction agents based on type.

    Maintains a mapping of chunk types to specialized agents and dispatches
    chunks to the correct handler.

    Attributes:
        _map: Dictionary mapping ChunkType to agent instance
    """

    def __init__(self, text: "TextAgent", table: "TableAgent", vision: "VisionAgent"):
        """
        Initialize router with extraction agents.

        Args:
            text: Agent for processing text chunks
            table: Agent for processing table chunks
            vision: Agent for processing figure/image chunks
        """
        self._map = {ChunkType.TEXT: text, ChunkType.TABLE: table, ChunkType.FIGURE: vision}

    def route(self, chunk: "RawChunk", trace: "TraceHandle | None" = None) -> "ProcessedChunk":
        """
        Route chunk to appropriate agent for processing.

        Args:
            chunk: Raw chunk to process
            trace: Optional Langfuse trace handle

        Returns:
            Processed chunk from appropriate agent
        """
        return self._map[chunk.chunk_type].process(chunk, trace=trace)
