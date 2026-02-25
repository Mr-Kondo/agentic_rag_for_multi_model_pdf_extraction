"""
Specialized extraction agents for different content types.

TextAgent, TableAgent, and VisionAgent process raw chunks using small
specialized models (2-4B) that stay loaded during ingestion.
"""

import json
import logging
from typing import TYPE_CHECKING

from mlx_lm import generate
from mlx_vlm import generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template

from src.agents.base import BaseAgent
from src.core.cache import _model_cache
from src.core.models import ChunkType, ProcessedChunk

if TYPE_CHECKING:
    from src.core.models import RawChunk
    from src.integrations.langfuse import TraceHandle

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════

_TEXT_SYSTEM = """You are a precise academic document analyst.
Given a text passage from a PDF, return ONLY valid JSON:
{
  "structured_text": "<cleaned passage>",
  "intuition_summary": "<1 sentence>",
  "key_concepts": ["<concept>"],
  "confidence": <0.0-1.0>,
  "agent_notes": "<issues>"
}"""

_TABLE_SYSTEM = """You are a structured-data extraction specialist.
Given a Markdown table, return ONLY valid JSON:
{
  "structured_text": "<corrected Markdown table>",
  "intuition_summary": "<1 sentence>",
  "key_concepts": ["<columns/metrics>"],
  "schema": {"columns": [], "row_count": 0, "units": {}},
  "confidence": <0.0-1.0>,
  "agent_notes": "<issues>"
}"""

_VISION_SYSTEM = """You are a scientific figure analyst.
Return ONLY valid JSON:
{
  "figure_type": "<bar_chart|line_chart|scatter_plot|flowchart|table_image|map|photograph|equation|network_diagram|other>",
  "structured_text": "<full description>",
  "intuition_summary": "<1 sentence>",
  "key_concepts": ["<labels>"],
  "confidence": <0.0-1.0>,
  "agent_notes": "<issues>"
}"""


# ═══════════════════════════════════════════════════════════
# TEXT AGENT
# ═══════════════════════════════════════════════════════════


class TextAgent(BaseAgent):
    """
    Extract structured data from text passages.

    Uses small text LLM (e.g., Phi-3.5-mini 3.8B) to clean and structure
    raw text from PDFs into searchable chunks.
    """

    def _load_model(self):
        """Load text model from cache."""
        self._model, self._tokenizer = _model_cache.load_text_model(self.model_id)

    def _run(self, chunk: "RawChunk", retry: bool = False, trace: "TraceHandle | None" = None) -> ProcessedChunk:
        """
        Extract structured data from text chunk.

        Args:
            chunk: Raw text chunk to process
            retry: Whether this is a retry attempt (adds retry prompt)
            trace: Optional Langfuse trace handle

        Returns:
            ProcessedChunk with structured text and metadata
        """
        content = str(chunk.raw_content) + (self.RETRY_SUFFIX if retry else "")
        messages = [
            {"role": "system", "content": _TEXT_SYSTEM},
            {"role": "user", "content": f"PASSAGE:\n{content}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Token measurement
        input_tokens = len(prompt) if isinstance(prompt, list) else len(self._tokenizer.encode(prompt))

        # Generation with tracing
        if trace:
            with trace.generation(
                name="text_extraction",
                model=self.model_id,
                input={"messages": messages},
                model_params={"max_tokens": 512},
            ) as g:
                raw = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=512, verbose=False)
                output_tokens = len(raw.split())
                g.set_output(raw, input_tokens=input_tokens, output_tokens=output_tokens)
        else:
            raw = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=512, verbose=False)

        p = self._safe_json(raw)
        return ProcessedChunk(
            chunk_type=ChunkType.TEXT,
            page_num=chunk.page_num,
            source_file=chunk.source_file,
            structured_text=p.get("structured_text", content[:2000]),
            intuition_summary=p.get("intuition_summary", ""),
            key_concepts=p.get("key_concepts", []),
            confidence=float(p.get("confidence", 0.7)),
            agent_notes=p.get("agent_notes", ""),
        )


# ═══════════════════════════════════════════════════════════
# TABLE AGENT
# ═══════════════════════════════════════════════════════════


class TableAgent(BaseAgent):
    """
    Extract structured data from markdown tables.

    Uses small text LLM (e.g., Qwen2.5-3B) to analyze and enhance
    markdown table representations with schema metadata.
    """

    def _load_model(self):
        """Load text model from cache."""
        self._model, self._tokenizer = _model_cache.load_text_model(self.model_id)

    def _run(self, chunk: "RawChunk", retry: bool = False, trace: "TraceHandle | None" = None) -> ProcessedChunk:
        """
        Extract structured data from table chunk.

        Args:
            chunk: Raw table (as markdown) to process
            retry: Whether this is a retry attempt
            trace: Optional Langfuse trace handle

        Returns:
            ProcessedChunk with enhanced table and schema metadata
        """
        content = str(chunk.raw_content) + (self.RETRY_SUFFIX if retry else "")
        messages = [
            {"role": "system", "content": _TABLE_SYSTEM},
            {"role": "user", "content": f"TABLE:\n{content}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Token measurement
        input_tokens = len(prompt) if isinstance(prompt, list) else len(self._tokenizer.encode(prompt))

        # Generation with tracing
        if trace:
            with trace.generation(
                name="table_extraction",
                model=self.model_id,
                input={"messages": messages},
                model_params={"max_tokens": 768},
            ) as g:
                raw = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=768, verbose=False)
                output_tokens = len(raw.split())
                g.set_output(raw, input_tokens=input_tokens, output_tokens=output_tokens)
        else:
            raw = generate(self._model, self._tokenizer, prompt=prompt, max_tokens=768, verbose=False)

        p = self._safe_json(raw)
        schema_ann = f"\n<!-- schema: {json.dumps(p.get('schema', {}), ensure_ascii=False)} -->"
        return ProcessedChunk(
            chunk_type=ChunkType.TABLE,
            page_num=chunk.page_num,
            source_file=chunk.source_file,
            structured_text=p.get("structured_text", content) + schema_ann,
            intuition_summary=p.get("intuition_summary", ""),
            key_concepts=p.get("key_concepts", []),
            confidence=float(p.get("confidence", 0.7)),
            agent_notes=p.get("agent_notes", ""),
        )


# ═══════════════════════════════════════════════════════════
# VISION AGENT
# ═══════════════════════════════════════════════════════════


class VisionAgent(BaseAgent):
    """
    Extract structured data from figures/images.

    Uses small vision-language model (e.g., SmolVLM-256M) to describe
    charts, diagrams, and other visual elements. Falls back to OCR if
    vision model fails to load.
    """

    def _load_model(self):
        """Load vision model from cache, fallback to OCR if unavailable."""
        try:
            self._model, self._processor, self._config = _model_cache.load_vision_model(self.model_id)
            self._use_vision = True
        except Exception as e:
            log.warning("VisionAgent: vision model failed (%s). OCR fallback.", e)
            self._use_vision = False

    def _run(self, chunk: "RawChunk", retry: bool = False, trace: "TraceHandle | None" = None) -> ProcessedChunk:
        """
        Extract structured data from figure chunk.

        Args:
            chunk: Raw image chunk to process
            retry: Whether this is a retry attempt
            trace: Optional Langfuse trace handle

        Returns:
            ProcessedChunk with figure description and metadata
        """
        if not self._use_vision or self._processor is None:
            return self._ocr_fallback(chunk)

        img = chunk.raw_content
        extra = self.RETRY_SUFFIX if retry else ""
        user_text = f"Describe.{extra}"

        # Combine system message with user text
        full_prompt = f"{_VISION_SYSTEM}\n\n{user_text}"
        try:
            prompt = apply_chat_template(self._processor, self._config, full_prompt, num_images=1)
        except (TypeError, AttributeError) as e:
            # Processor is malformed or None - use OCR fallback
            log.warning(f"Vision model processor error: {e}. Using OCR fallback.")
            return self._ocr_fallback(chunk)

        # Generation with tracing (VLM - token count not available)
        try:
            if trace:
                with trace.generation(
                    name="vision_extraction",
                    model=self.model_id,
                    input={"prompt": full_prompt, "has_image": True},
                    model_params={},
                ) as g:
                    result = vlm_generate(self._model, self._processor, prompt, [img], verbose=False)
                    # Extract text from GenerationResult object
                    output = result if isinstance(result, str) else str(result)
                    # Token count not available for VLM models
                    g.set_output(output, input_tokens=None, output_tokens=None)
            else:
                result = vlm_generate(self._model, self._processor, prompt, [img], verbose=False)
                output = result if isinstance(result, str) else str(result)
        except (TypeError, AttributeError, RuntimeError) as e:
            # Vision generation failed - use OCR fallback
            log.warning(f"Vision generation error: {e}. Using OCR fallback.")
            return self._ocr_fallback(chunk)

        p = self._safe_json(output)
        return ProcessedChunk(
            chunk_type=ChunkType.FIGURE,
            page_num=chunk.page_num,
            source_file=chunk.source_file,
            structured_text=p.get("structured_text", output[:1000]),
            intuition_summary=p.get("intuition_summary", ""),
            key_concepts=p.get("key_concepts", []),
            confidence=float(p.get("confidence", 0.6)),
            agent_notes=f"figure_type={p.get('figure_type', '?')} | {p.get('agent_notes', '')}",
        )

    def _ocr_fallback(self, chunk: "RawChunk") -> ProcessedChunk:
        """
        Fallback to OCR if vision model unavailable.

        Args:
            chunk: Raw image chunk

        Returns:
            ProcessedChunk with OCR-extracted text (low confidence)
        """
        try:
            import pytesseract

            text = pytesseract.image_to_string(chunk.raw_content)
        except Exception:
            text = "[OCR unavailable]"
        return ProcessedChunk(
            chunk_type=ChunkType.FIGURE,
            page_num=chunk.page_num,
            source_file=chunk.source_file,
            structured_text=text,
            intuition_summary="OCR fallback.",
            confidence=0.3,
            agent_notes="Vision model not loaded.",
        )
