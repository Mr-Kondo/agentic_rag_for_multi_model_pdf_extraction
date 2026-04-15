"""
Specialized extraction agents for different content types.

TextAgent, TableAgent, and VisionAgent process raw chunks by calling
Ollama-hosted models via the Ollama Python client.
"""

import io
import json
import logging
from typing import TYPE_CHECKING

import ollama

from src.agents.base import BaseAgent
from src.agents.table_extraction import TableFromImageExtractor
from src.core.cache import _model_cache
from src.core.config import config
from src.core.models import ChunkType, ProcessedChunk
from src.utils.token_counter import count_tokens

if TYPE_CHECKING:
    from src.core.models import RawChunk, RegionOCRPolicy
    from src.integrations.langfuse import TraceHandle

log = logging.getLogger(__name__)


def _is_structured_table_markdown(table_markdown: str | None) -> bool:
    """Return True when markdown has a minimally valid table structure."""
    if not table_markdown:
        return False

    lines = [line.strip() for line in table_markdown.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    header_line = lines[0]
    separator_line = lines[1]
    if "|" not in header_line or "|" not in separator_line or "---" not in separator_line:
        return False

    header_cells = [cell.strip() for cell in header_line.strip("|").split("|")]
    return len(header_cells) >= 2


def _chunk_kwargs(chunk: "RawChunk") -> dict:
    """Copy provenance metadata from RawChunk into ProcessedChunk kwargs."""
    return {
        "page_num": chunk.page_num,
        "source_file": chunk.source_file,
        "bbox": chunk.bbox,
        "page_width": chunk.page_width,
        "page_height": chunk.page_height,
        "artifact_path": chunk.artifact_path,
        "source_preview": chunk.source_preview,
    }


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
}
Preserve the original language exactly as written.
Do not translate or convert Japanese text into any other language.
Do not convert Japanese kanji (漢字) into Simplified or Traditional Chinese characters.
Keep all Japanese characters (hiragana, katakana, kanji) unchanged.
日本語のテキストは必ずそのまま保持し、中国語（簡体字・繁体字）に変換しないでください。"""

_OCR_LANG = str(config.get("ocr.default_lang", "jpn+eng"))

_TABLE_SYSTEM = """You are a structured-data extraction specialist.
Given a Markdown table, return ONLY valid JSON:
{
  "structured_text": "<corrected Markdown table>",
  "intuition_summary": "<1 sentence>",
  "key_concepts": ["<columns/metrics>"],
  "schema": {"columns": [], "row_count": 0, "units": {}},
  "confidence": <0.0-1.0>,
  "agent_notes": "<issues>"
}
Preserve the original language of all cell values exactly as written.
Do not convert Japanese kanji into Simplified or Traditional Chinese characters.
日本語テキストはそのまま保持し、中国語に変換しないでください。"""

_VISION_SYSTEM = """You are a scientific figure analyst.
Carefully identify the type of visual content. If you detect a structured table in the image, classify it as 'table_image' with HIGH confidence (0.8+).
Return ONLY valid JSON:
{
  "figure_type": "<bar_chart|line_chart|scatter_plot|flowchart|table_image|map|photograph|equation|network_diagram|other>",
  "structured_text": "<full description>",
  "intuition_summary": "<1 sentence>",
  "key_concepts": ["<labels>"],
  "confidence": <0.0-1.0>,
  "agent_notes": "<issues>"
}
Preserve any Japanese text exactly as written. Do not convert to Chinese.
日本語テキストはそのまま保持し、中国語に変換しないでください。"""


# ═══════════════════════════════════════════════════════════
# TEXT AGENT
# ═══════════════════════════════════════════════════════════


class TextAgent(BaseAgent):
    """
    Extract structured data from text passages.

    Calls an Ollama-hosted text model to clean and structure raw text
    from PDFs into searchable chunks.
    """

    def _load_model(self):
        """Obtain Ollama client for text inference."""
        self._client: ollama.Client = _model_cache.load_text_model(self.model_id)

    def _run(
        self,
        chunk: "RawChunk",
        retry: bool = False,
        trace: "TraceHandle | None" = None,
        policy: "RegionOCRPolicy | None" = None,
    ) -> ProcessedChunk:
        """
        Extract structured data from text chunk.

        Args:
            chunk: Raw text chunk to process
            retry: Whether this is a retry attempt (adds retry prompt)
            trace: Optional Langfuse trace handle

        Returns:
            ProcessedChunk with structured text and metadata
        """
        if bool(config.get("pipeline.text_passthrough", True)):
            raw_text = str(chunk.raw_content).strip()
            summary = raw_text.replace("\n", " ")[:200]
            return ProcessedChunk(
                chunk_type=ChunkType.TEXT,
                **_chunk_kwargs(chunk),
                structured_text=raw_text,
                intuition_summary=summary,
                key_concepts=[],
                confidence=0.95,
                agent_notes="text_passthrough",
            )

        content = str(chunk.raw_content) + (self.RETRY_SUFFIX if retry else "")
        messages = [
            {"role": "system", "content": _TEXT_SYSTEM},
            {"role": "user", "content": f"PASSAGE:\n{content}"},
        ]

        input_tokens = count_tokens(content)

        if trace:
            with trace.generation(
                name="text_extraction",
                model=self.model_id,
                input={"messages": messages},
                model_params={"max_tokens": 512},
            ) as g:
                response = self._client.chat(
                    model=self.model_id,
                    messages=messages,
                    options={"num_predict": 512, "temperature": 0.0},
                    stream=False,
                )
                raw = response.message.content
                output_tokens = response.eval_count or count_tokens(raw)
                g.set_output(raw, input_tokens=input_tokens, output_tokens=output_tokens)
        else:
            response = self._client.chat(
                model=self.model_id,
                messages=messages,
                options={"num_predict": 512, "temperature": 0.0},
                stream=False,
            )
            raw = response.message.content

        p = self._safe_json(raw)
        return ProcessedChunk(
            chunk_type=ChunkType.TEXT,
            **_chunk_kwargs(chunk),
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

    Calls an Ollama-hosted text model to analyze and enhance markdown table
    representations with schema metadata.
    """

    def _load_model(self):
        """Obtain Ollama client for text inference."""
        self._client: ollama.Client = _model_cache.load_text_model(self.model_id)

    def _run(
        self,
        chunk: "RawChunk",
        retry: bool = False,
        trace: "TraceHandle | None" = None,
        policy: "RegionOCRPolicy | None" = None,
    ) -> ProcessedChunk:
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

        input_tokens = count_tokens(content)

        if trace:
            with trace.generation(
                name="table_extraction",
                model=self.model_id,
                input={"messages": messages},
                model_params={"max_tokens": 768},
            ) as g:
                response = self._client.chat(
                    model=self.model_id,
                    messages=messages,
                    options={"num_predict": 768, "temperature": 0.0},
                    stream=False,
                )
                raw = response.message.content
                output_tokens = response.eval_count or count_tokens(raw)
                g.set_output(raw, input_tokens=input_tokens, output_tokens=output_tokens)
        else:
            response = self._client.chat(
                model=self.model_id,
                messages=messages,
                options={"num_predict": 768, "temperature": 0.0},
                stream=False,
            )
            raw = response.message.content

        p = self._safe_json(raw)
        schema_ann = f"\n<!-- schema: {json.dumps(p.get('schema', {}), ensure_ascii=False)} -->"
        return ProcessedChunk(
            chunk_type=ChunkType.TABLE,
            **_chunk_kwargs(chunk),
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

    Calls an Ollama-hosted vision-language model to describe charts, diagrams,
    and other visual elements. Falls back to OCR if the Ollama server is
    unavailable or the model cannot process the image.
    """

    def _load_model(self):
        """Obtain vision inference client (Ollama or vLLM)."""
        try:
            client = _model_cache.load_vision_model(self.model_id)
            self._client = client
            self._is_vllm = "/" in self.model_id  # HF hub IDs contain '/'
            self._use_vision = True
            log.info(f"VisionAgent: loaded model {self.model_id} ({'vLLM' if self._is_vllm else 'Ollama'})")
        except Exception as e:
            log.warning("VisionAgent: client unavailable (%s). OCR fallback.", e)
            self._client = None
            self._use_vision = False
            self._is_vllm = False

    def _run(
        self,
        chunk: "RawChunk",
        retry: bool = False,
        trace: "TraceHandle | None" = None,
        policy: "RegionOCRPolicy | None" = None,
    ) -> ProcessedChunk:
        """
        Extract structured data from figure chunk.

        Args:
            chunk: Raw image chunk to process
            retry: Whether this is a retry attempt
            trace: Optional Langfuse trace handle

        Returns:
            ProcessedChunk with figure description and metadata
        """
        if bool(config.get("pipeline.figure_ocr_only", True)):
            return self._ocr_fallback(chunk)

        table_extractor = TableFromImageExtractor(policy=policy)
        if table_extractor.is_probable_table_image(chunk.raw_content):
            log.info("VisionAgent: geometry heuristic detected table-like image on page %d", chunk.page_num)
            table_markdown = table_extractor.extract_table_from_image(chunk.raw_content, policy=policy)
            if _is_structured_table_markdown(table_markdown):
                return ProcessedChunk(
                    chunk_type=ChunkType.TABLE,
                    **_chunk_kwargs(chunk),
                    structured_text=table_markdown,
                    intuition_summary="Extracted from table-like image via geometry heuristic.",
                    key_concepts=[],
                    confidence=0.75,
                    agent_notes="extracted_from_geometry_heuristic",
                )

        if not self._use_vision or self._client is None:
            return self._ocr_fallback(chunk)

        img = chunk.raw_content
        extra = self.RETRY_SUFFIX if retry else ""
        user_text = f"Describe.{extra}"
        full_prompt = f"{_VISION_SYSTEM}\n\n{user_text}"

        # Encode PIL image as PNG bytes for Ollama
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        try:
            if trace:
                input_tokens = count_tokens(full_prompt) + 256  # 256 = typical image patch budget
                with trace.generation(
                    name="vision_extraction",
                    model=self.model_id,
                    input={"prompt": full_prompt, "has_image": True},
                    model_params={},
                ) as g:
                    output = self._infer(image_bytes, full_prompt)
                    output_tokens = count_tokens(output)
                    g.set_output(output, input_tokens=input_tokens, output_tokens=output_tokens)
            else:
                output = self._infer(image_bytes, full_prompt)
        except Exception as e:
            log.warning("VisionAgent: inference error (%s). OCR fallback.", e)
            return self._ocr_fallback(chunk)

        p = self._safe_json(output)
        figure_type = p.get("figure_type", "other")
        confidence = float(p.get("confidence", 0.6))
        log.info(
            "VisionAgent: page=%d figure_type=%s confidence=%.2f",
            chunk.page_num,
            figure_type,
            confidence,
        )

        # Check if this is a table image - attempt to extract table structure
        if figure_type == "table_image" and confidence >= 0.4:
            try:
                table_markdown = table_extractor.extract_table_from_image(chunk.raw_content, policy=policy)
                if _is_structured_table_markdown(table_markdown):
                    log.debug("VisionAgent: table image extraction successful (%s)", chunk.source_file)
                    return ProcessedChunk(
                        chunk_type=ChunkType.TABLE,
                        **_chunk_kwargs(chunk),
                        structured_text=table_markdown,
                        intuition_summary=p.get("intuition_summary", ""),
                        key_concepts=p.get("key_concepts", []),
                        confidence=confidence,
                        agent_notes=f"extracted_from_image | {p.get('agent_notes', '')}",
                    )
            except Exception as e:
                log.debug("VisionAgent: table extraction failed: %s. Falling back to FIGURE.", e)

        # Last-chance rescue: try deterministic table extraction for figure chunks.
        try:
            rescued_markdown = table_extractor.extract_table_from_image(chunk.raw_content, policy=policy)
            if _is_structured_table_markdown(rescued_markdown):
                log.info("VisionAgent: rescued table extraction on page %d", chunk.page_num)
                return ProcessedChunk(
                    chunk_type=ChunkType.TABLE,
                    **_chunk_kwargs(chunk),
                    structured_text=rescued_markdown,
                    intuition_summary="Rescued from figure via deterministic table extraction.",
                    key_concepts=p.get("key_concepts", []),
                    confidence=max(0.55, min(confidence, 0.75)),
                    agent_notes=f"rescued_table_from_figure | figure_type={figure_type}",
                )
        except Exception as e:
            log.debug("VisionAgent: rescue extraction failed: %s", e)

        # Default: return as FIGURE chunk
        return ProcessedChunk(
            chunk_type=ChunkType.FIGURE,
            **_chunk_kwargs(chunk),
            structured_text=p.get("structured_text", output[:1000]),
            intuition_summary=p.get("intuition_summary", ""),
            key_concepts=p.get("key_concepts", []),
            confidence=confidence,
            agent_notes=f"figure_type={figure_type} | {p.get('agent_notes', '')}",
        )

    def _infer(self, img_bytes: bytes, full_prompt: str) -> str:
        """
        Run inference on image + prompt via Ollama or vLLM.

        Args:
            img_bytes: PNG-encoded image bytes
            full_prompt: Text prompt with system message

        Returns:
            Model output text

        Raises:
            Exception: If inference fails
        """
        if self._is_vllm:
            return self._infer_vllm(img_bytes, full_prompt)
        else:
            return self._infer_ollama(img_bytes, full_prompt)

    def _infer_ollama(self, img_bytes: bytes, full_prompt: str) -> str:
        """
        Inference via Ollama chat API.

        Args:
            img_bytes: PNG bytes
            full_prompt: Prompt with system message

        Returns:
            Model output text
        """
        messages = [
            {"role": "user", "content": full_prompt, "images": [img_bytes]},
        ]
        response = self._client.chat(
            model=self.model_id,
            messages=messages,
            options={"num_predict": 512, "temperature": 0.0},
            stream=False,
        )
        return response.message.content

    def _infer_vllm(self, img_bytes: bytes, full_prompt: str) -> str:
        """
        Inference via vLLM.

        Processes image + prompt using vLLM LLM.generate API.

        Args:
            img_bytes: PNG bytes
            full_prompt: Prompt with system message

        Returns:
            Model output text

        Raises:
            ImportError: If required dependencies unavailable
        """
        try:
            from transformers import AutoProcessor
            from qwen_vl_utils import process_vision_info
            from vllm import SamplingParams
            from PIL import Image
        except ImportError as e:
            raise ImportError("vLLM inference requires: pip install transformers qwen-vl-utils vllm") from e

        # Convert PNG bytes to PIL image
        img = Image.open(io.BytesIO(img_bytes))

        # Prepare messages for vLLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": full_prompt},
                ],
            },
        ]

        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            }

            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
            )

            outputs = self._client.generate([llm_inputs], sampling_params=sampling_params)
            return outputs[0].outputs[0].text

        except Exception as e:
            log.warning(f"vLLM inference failed: {e}")
            raise

    def _ocr_fallback(self, chunk: "RawChunk") -> ProcessedChunk:
        """
        Fallback to OCR if vision model unavailable.

        Args:
            chunk: Raw image chunk

        Returns:
            ProcessedChunk with OCR-extracted text (low confidence)
        """
        text = ""
        try:
            import pytesseract

            try:
                text = pytesseract.image_to_string(chunk.raw_content, lang=_OCR_LANG)
            except Exception:
                fallback_lang = str(config.get("ocr.fallback_lang", "eng"))
                text = pytesseract.image_to_string(chunk.raw_content, lang=fallback_lang)
        except Exception:
            text = ""

        if not text.strip():
            text = "[OCR unavailable]"

        return ProcessedChunk(
            chunk_type=ChunkType.FIGURE,
            **_chunk_kwargs(chunk),
            structured_text=text,
            intuition_summary="OCR fallback.",
            confidence=0.3,
            agent_notes="Vision model not loaded.",
        )
