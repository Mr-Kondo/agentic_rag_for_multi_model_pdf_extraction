"""
Reasoning orchestrator agent for RAG answer generation.

Uses large reasoning model (8B+) with explicit load/unload lifecycle
to generate answers grounded in retrieved context.
"""

import logging
import re
from typing import TYPE_CHECKING

from mlx_lm import generate

from src.agents.base import BaseLoadableModel
from src.core.cache import _model_cache
from src.core.models import ChunkType, RAGAnswer

if TYPE_CHECKING:
    from src.core.store import ChunkStore
    from src.integrations.langfuse import TraceHandle

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

_ORCHESTRATOR_SYSTEM = """You are a research assistant with deep reasoning capability.
Think step-by-step inside <think> tags, then write your final answer.
Your final answer must:
  - Be grounded ONLY in the retrieved context.
  - Cite source_file and page_num for every claim.
  - Note when information comes from a figure description.
  - State "Insufficient context" if context is insufficient.

Retrieved context:
{context}

Question:
{question}
"""

_VISUAL_KEYWORDS = {
    "figure",
    "graph",
    "chart",
    "flow",
    "diagram",
    "image",
    "plot",
    "map",
    "図",
    "グラフ",
    "フロー",
    "フローチャート",
    "チャート",
    "表",
}


# ═══════════════════════════════════════════════════════════
# REASONING ORCHESTRATOR AGENT
# ═══════════════════════════════════════════════════════════


class ReasoningOrchestratorAgent(BaseLoadableModel):
    """
    Large reasoning model for RAG answer generation.

    Implements BaseLoadableModel for explicit load/unload lifecycle.
    Separates retrieval (no model needed) from generation (model required)
    to minimize VRAM usage.

    The split design enables:
      1. Retrieve chunks (no model loaded → no VRAM)
      2. Load orchestrator (~4GB)
      3. Generate answer
      4. Unload orchestrator (frees VRAM before answer validation)

    Attributes:
        _THINK_RE: Regex pattern for extracting <think> reasoning blocks
    """

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

    def _do_load(self) -> None:
        """Load reasoning model and tokenizer."""
        self._model, self._tokenizer = _model_cache.load_text_model(self.model_id)

    def _do_unload(self) -> None:
        """Unload model and tokenizer references."""
        del self._model
        del self._tokenizer

    # ── Retrieval (no model needed) ────────────────────────

    def retrieve(
        self,
        question: str,
        store: "ChunkStore",
        trace: "TraceHandle | None" = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks from vector store.

        Performs semantic search and optionally adds figure-specific results
        if visual keywords detected in question. Does not require model to
        be loaded.

        Args:
            question: User's question
            store: ChunkStore for vector search
            trace: Optional Langfuse trace handle

        Returns:
            List of retrieved chunk dicts with text, metadata, and scores
        """

        def _do():
            hits = store.query(question, n_results=8)
            # Add figure-specific search if visual keywords present
            if any(kw in question.lower() for kw in _VISUAL_KEYWORDS):
                fig = store.query(question, n_results=3, chunk_type=ChunkType.FIGURE)
                seen = {h["text"] for h in hits}
                hits += [h for h in fig if h["text"] not in seen]
            return hits

        if trace:
            with trace.span("retrieve_chunks", input={"question": question[:200]}) as s:
                hits = _do()
                s.update(output={"n_hits": len(hits)})
        else:
            hits = _do()
        return hits

    # ── Generation (model required) ───────────────────────

    def generate(
        self,
        question: str,
        hits: list[dict],
        trace: "TraceHandle | None" = None,
    ) -> RAGAnswer:
        """
        Generate RAG answer from retrieved chunks.

        Requires model to be loaded. Uses retrieved chunks as context
        and generates grounded answer with reasoning trace.

        Args:
            question: User's question
            hits: Retrieved chunks from vector store
            trace: Optional Langfuse trace handle

        Returns:
            RAGAnswer with answer text, reasoning, and source references

        Raises:
            RuntimeError: If model is not loaded
        """
        self._assert_loaded()
        context_str = self._build_context(hits)
        prompt = _ORCHESTRATOR_SYSTEM.format(context=context_str, question=question)
        messages = [{"role": "user", "content": prompt}]

        formatted_prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Token measurement
        input_tokens = (
            len(formatted_prompt) if isinstance(formatted_prompt, list) else len(self._tokenizer.encode(formatted_prompt))
        )

        # Generation with tracing
        if trace:
            with trace.generation(
                name="orchestrator_reasoning",
                model=self.model_id,
                input={"messages": messages},
                model_params={"max_tokens": 2048},
            ) as g:
                output = generate(self._model, self._tokenizer, prompt=formatted_prompt, max_tokens=2048, verbose=False)
                output_tokens = len(output.split())
                g.set_output(output, input_tokens=input_tokens, output_tokens=output_tokens)
        else:
            output = generate(self._model, self._tokenizer, prompt=formatted_prompt, max_tokens=2048, verbose=False)

        reasoning, answer = self._strip_reasoning(output)
        return RAGAnswer(
            question=question,
            answer=answer,
            reasoning_trace=reasoning,
            source_chunks=[
                {
                    "type": h["meta"]["chunk_type"],
                    "file": h["meta"]["source_file"],
                    "page": h["meta"]["page_num"],
                    "score": round(h["score"], 3),
                    "summary": h["meta"]["intuition_summary"],
                    "text": h["text"],
                }
                for h in hits
            ],
        )

    def _strip_reasoning(self, raw: str) -> tuple[str, str]:
        """
        Extract reasoning from <think> tags and separate from answer.

        Args:
            raw: Raw model output potentially containing <think> blocks

        Returns:
            Tuple of (reasoning_text, answer_text)
        """
        match = self._THINK_RE.search(raw)
        if match:
            return match.group(1).strip(), self._THINK_RE.sub("", raw).strip()
        return "", raw.strip()

    @staticmethod
    def _build_context(hits: list[dict]) -> str:
        """
        Format retrieved chunks into context string for prompt.

        Args:
            hits: List of retrieved chunk dicts

        Returns:
            Formatted context string with numbered chunks
        """
        parts = []
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            parts.append(
                f"[{i}] ({m['chunk_type'].upper()} | {m['source_file']} p.{m['page_num']} | "
                f"score={h['score']:.2f})\n"
                f"Summary: {m['intuition_summary']}\nContent: {h['text'][:800]}"
            )
        return "\n\n---\n\n".join(parts)
