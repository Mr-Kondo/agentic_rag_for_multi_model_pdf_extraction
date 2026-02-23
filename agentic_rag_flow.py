"""
agentic_rag_flow_v3.py
======================
Changes from v2:
  1. ReasoningOrchestratorAgent gains BaseLoadableModel:
       .load() / .unload() / context manager
       → only loaded during generation, unloaded immediately after

  2. ValidatorAgent split into ChunkValidatorAgent + AnswerValidatorAgent
       (see validator_agent.py)

  3. Sequential load/unload order in the pipeline:

     ingest():
       ┌─ parse (no LLM) ──────────────────────────────────┐
       │  for each chunk:                                   │
       │    agent.process(chunk)    ← 3 small SLMs         │
       │  [all small SLMs already loaded at build time]    │
       │                                                    │
       │  ChunkValidatorAgent.load()                        │
       │  for each chunk:                                   │
       │    chunk_validator.validate_chunk(raw, processed)  │
       │  ChunkValidatorAgent.unload()   ← freed here      │
       └────────────────────────────────────────────────────┘

     query():
       ┌─ retrieve (ChunkStore, no LLM) ───────────────────┐
       │                                                    │
       │  OrchestratorAgent.load()                         │
       │  answer = orchestrator.generate(question, hits)   │
       │  OrchestratorAgent.unload()  ← freed here        │
       │                                                    │
       │  AnswerValidatorAgent.load()                      │
       │  val = answer_validator.validate_answer(...)      │
       │  AnswerValidatorAgent.unload() ← freed here      │
       └────────────────────────────────────────────────────┘

  4. VRAM high-water mark: max(orchestrator, answer_validator) ≈ 16 GB
     (never two ~10B models simultaneously)

  5. The 3 small SLMs (text/table/vision agents) remain loaded throughout
     ingest — they are 2-4B each and loaded at pipeline.build() time.
     If VRAM is very tight, set lazy_agents=True in build() to load/unload
     them per-chunk (see _route_lazy).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import chromadb
import pdfplumber
import pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer

from mlx_lm import generate, load
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from langfuse_tracer import LangfuseTracer, _TraceHandle
from validator_agent import (
    AnswerValidationResult,
    AnswerValidatorAgent,
    BaseLoadableModel,
    ChunkValidationResult,
    ChunkValidatorAgent,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ═══════════════════════════════════════════════════════════
# 0. MODEL CACHE SYSTEM
# ═══════════════════════════════════════════════════════════

MODEL_CACHE_DIR = Path.home() / ".models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configure HuggingFace to cache models locally
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR.resolve())


class ModelCache:
    """
    Manages in-memory caching of loaded models and tracks their usage.
    Models are cached in memory while loaded to avoid redundant downloads.
    Cleanup removes models from .models/ that are no longer being used.
    """

    def __init__(self):
        self._text_models: dict[str, Any] = {}  # {model_id: loaded_model}
        self._vision_models: dict[str, Any] = {}  # {model_id: loaded_model}
        self._model_usage: set[str] = set()  # Track which models are currently in use
        self._lock = None  # Could be threading.Lock() for thread safety if needed

    def load_text_model(self, model_id: str) -> Any:
        """Load a text model from cache or download it."""
        if model_id in self._text_models:
            log.debug(f"📦 Returning cached text model: {model_id}")
            return self._text_models[model_id]

        log.info(f"🔄 Loading text model: {model_id}")
        model = load(model_id)
        self._text_models[model_id] = model
        self._model_usage.add(model_id)
        return model

    def load_vision_model(self, model_id: str) -> tuple[Any, Any, Any]:
        """Load a vision model from cache or download it. Returns (model, processor, config)."""
        if model_id in self._vision_models:
            log.debug(f"📦 Returning cached vision model: {model_id}")
            return self._vision_models[model_id]

        log.info(f"🔄 Loading vision model: {model_id}")
        model, processor = vlm_load(model_id)
        config = load_config(model_id)
        cached_model = (model, processor, config)
        self._vision_models[model_id] = cached_model
        self._model_usage.add(model_id)
        return cached_model

    def cleanup_unused_models(self):
        """
        Remove model directories from .models/ that are not currently loaded.
        This frees up disk space after models are no longer needed.
        """
        try:
            if not MODEL_CACHE_DIR.exists():
                return

            # Get list of cached model directories
            cached_models = set(d.name for d in MODEL_CACHE_DIR.glob("**/") if d.is_dir())

            # Find models not currently in use
            loaded_models = set(self._text_models.keys()) | set(self._vision_models.keys())
            unused = cached_models - self._model_usage

            if unused:
                log.info(f"🧹 Cleaning up {len(unused)} unused model(s)...")
                for model_name in unused:
                    model_path = MODEL_CACHE_DIR / model_name
                    if model_path.exists():
                        log.info(f"  Removing {model_name}...")
                        shutil.rmtree(model_path, ignore_errors=True)
        except Exception as e:
            log.warning(f"⚠️  Error during model cleanup: {e}")


# Global model cache instance
_model_cache = ModelCache()


# ═══════════════════════════════════════════════════════════
# OUTPUT UTILITIES
# ═══════════════════════════════════════════════════════════

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def serialize_chunk(chunk: "ProcessedChunk") -> dict:
    """Convert ProcessedChunk to JSON-serializable dict."""
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type.value,
        "page_num": chunk.page_num,
        "source_file": chunk.source_file,
        "structured_text": chunk.structured_text[:500] + "..." if len(chunk.structured_text) > 500 else chunk.structured_text,
        "intuition_summary": chunk.intuition_summary,
        "key_concepts": chunk.key_concepts,
        "confidence": chunk.confidence,
        "agent_notes": chunk.agent_notes,
        "validation": {
            "verdict_score": chunk.validation.verdict_score,
            "issues": chunk.validation.issues,
            "corrected_text": chunk.validation.corrected.structured_text[:500] + "..."
            if chunk.validation.corrected and len(chunk.validation.corrected.structured_text or "") > 500
            else (chunk.validation.corrected.structured_text if chunk.validation.corrected else None),
        }
        if chunk.validation
        else None,
    }


def save_chunks(chunks: list["ProcessedChunk"], pdf_name: str) -> None:
    """Save processed chunks to JSON file."""
    chunks_data = [serialize_chunk(c) for c in chunks]
    output_path = OUTPUT_DIR / f"{Path(pdf_name).stem}_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    log.info(f"✓ Saved {len(chunks)} chunks to {output_path}")


def save_answer(result: "RAGAnswer", pdf_name: str, question: str) -> None:
    """Save RAG answer to JSON file."""
    answer_data = {
        "pdf_file": pdf_name,
        "question": question,
        "answer": result.answer,
        "reasoning_trace": result.reasoning_trace[:1000] + "..."
        if len(result.reasoning_trace) > 1000
        else result.reasoning_trace,
        "source_chunks": result.source_chunks,
        "trace_id": result.trace_id,
        "validation": {
            "answer_is_grounded": result.validation_summary.answer_is_grounded,
            "hallucinations": result.validation_summary.hallucinations,
            "answer_verdict_score": result.validation_summary.answer_verdict_score,
            "validator_notes": result.validation_summary.validator_notes,
            "answer_was_revised": result.validation_summary.answer_was_revised,
        }
        if result.validation_summary
        else None,
    }

    output_path = OUTPUT_DIR / f"{Path(pdf_name).stem}_answer.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_data, f, ensure_ascii=False, indent=2)

    log.info(f"✓ Saved answer to {output_path}")


# 1. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class RawChunk:
    chunk_type: ChunkType
    page_num: int
    raw_content: Any
    bbox: tuple | None = None
    source_file: str = ""


@dataclass
class ProcessedChunk:
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_type: ChunkType = ChunkType.TEXT
    page_num: int = 0
    source_file: str = ""
    structured_text: str = ""
    intuition_summary: str = ""
    key_concepts: list[str] = field(default_factory=list)
    confidence: float = 1.0
    agent_notes: str = ""
    embedding: list[float] = field(default_factory=list)
    validation: ChunkValidationResult | None = None


@dataclass
class ValidationSummary:
    answer_is_grounded: bool
    hallucinations: list[str]
    answer_verdict_score: float
    validator_notes: str
    answer_was_revised: bool


@dataclass
class RAGAnswer:
    question: str
    answer: str
    reasoning_trace: str
    source_chunks: list[dict] = field(default_factory=list)
    trace_id: str = ""
    validation_summary: ValidationSummary | None = None


# ═══════════════════════════════════════════════════════════
# 2. PDF PARSER  (no LLM — always in memory, lightweight)
# ═══════════════════════════════════════════════════════════


class PDFParser:
    MIN_TABLE_ROWS = 2
    MIN_TEXT_LEN = 40

    def parse(self, pdf_path: str | Path) -> list[RawChunk]:
        pdf_path = Path(pdf_path)
        chunks: list[RawChunk] = []
        doc_fitz = pymupdf.open(str(pdf_path))
        doc_plumb = pdfplumber.open(str(pdf_path))

        for page_idx in range(len(doc_fitz)):
            fitz_page = doc_fitz[page_idx]
            plumb_page = doc_plumb.pages[page_idx]

            for table in plumb_page.extract_tables():
                if table and len(table) >= self.MIN_TABLE_ROWS:
                    chunks.append(
                        RawChunk(
                            chunk_type=ChunkType.TABLE,
                            page_num=page_idx + 1,
                            raw_content=self._to_markdown(table),
                            source_file=pdf_path.name,
                        )
                    )

            for img_info in fitz_page.get_images(full=True):
                try:
                    xref = img_info[0]
                    pix = pymupdf.Pixmap(doc_fitz, xref)

                    # Convert to RGB if not already
                    if pix.n > 4:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    elif pix.n == 4:
                        # RGBA - convert to RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    elif pix.n not in (3, 4):
                        # Other color spaces - convert to RGB
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                    # Skip very small images
                    if pix.width < 80 or pix.height < 80:
                        continue

                    # Create PIL Image from pixmap data with fallback handling
                    try:
                        # First try with raw samples (works for properly formatted pixmaps)
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    except ValueError:
                        # Fallback: use numpy array conversion if available
                        try:
                            import numpy as np

                            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                            # Reshape to proper dimensions
                            expected_size = pix.width * pix.height * 3
                            if len(img_array) >= expected_size:
                                img_array = img_array[:expected_size].reshape((pix.height, pix.width, 3))
                                img = Image.fromarray(img_array, "RGB")
                            else:
                                log.warning(
                                    f"Image on page {page_idx + 1} has insufficient data "
                                    f"({len(img_array)} bytes, expected {expected_size}). Skipping."
                                )
                                continue
                        except ImportError:
                            # numpy not available - skip this image
                            log.warning(f"Cannot extract image from page {page_idx + 1} (numpy not available). Skipping.")
                            continue

                    chunks.append(
                        RawChunk(
                            chunk_type=ChunkType.FIGURE,
                            page_num=page_idx + 1,
                            raw_content=img,
                            source_file=pdf_path.name,
                        )
                    )
                except Exception as e:
                    log.warning(f"Error extracting image from page {page_idx + 1}: {e}. Skipping.")
                    continue

            raw_text = plumb_page.extract_text() or ""
            if len(raw_text.strip()) >= self.MIN_TEXT_LEN:
                chunks.append(
                    RawChunk(
                        chunk_type=ChunkType.TEXT,
                        page_num=page_idx + 1,
                        raw_content=raw_text,
                        source_file=pdf_path.name,
                    )
                )

        doc_fitz.close()
        doc_plumb.close()
        log.info("Parsed %d raw chunks from %s", len(chunks), pdf_path.name)
        return chunks

    @staticmethod
    def _to_markdown(table: list[list]) -> str:
        if not table:
            return ""
        header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
        sep = "| " + " | ".join("---" for _ in table[0]) + " |"
        rows = ["| " + " | ".join(str(c or "") for c in row) + " |" for row in table[1:]]
        return "\n".join([header, sep] + rows)


# ═══════════════════════════════════════════════════════════
# 3. BASE AGENT  (small SLMs — kept loaded during ingest)
# ═══════════════════════════════════════════════════════════


class BaseAgent:
    CONFIDENCE_THRESHOLD = 0.5
    RETRY_SUFFIX = "\n[RETRY] Low confidence. Be conservative; flag unknowns explicitly."

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def process(self, chunk: RawChunk, trace: _TraceHandle | None = None) -> ProcessedChunk:
        if trace:
            with trace.span(
                f"agent_{chunk.chunk_type.value}",
                input={"page": chunk.page_num},
            ) as s:
                result = self._run_with_retry(chunk, trace)
                s.update(output={"confidence": result.confidence})
        else:
            result = self._run_with_retry(chunk, None)
        return result

    def _run_with_retry(self, chunk: RawChunk, trace: _TraceHandle | None) -> ProcessedChunk:
        result = self._run(chunk, retry=False, trace=trace)
        if result.confidence < self.CONFIDENCE_THRESHOLD:
            log.warning("%s: retrying p.%d (conf=%.2f)", self.__class__.__name__, chunk.page_num, result.confidence)
            result = self._run(chunk, retry=True, trace=trace)
        return result

    def _run(self, chunk: RawChunk, retry: bool = False, trace: _TraceHandle | None = None) -> ProcessedChunk:
        raise NotImplementedError

    @staticmethod
    def _safe_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _last_content(output: Any) -> str:
        if isinstance(output, list) and output:
            last = output[-1]
            if isinstance(last, dict):
                return last.get("content", str(last))
        return str(output)


# ═══════════════════════════════════════════════════════════
# 4–6. TEXT / TABLE / VISION AGENTS  (small, always loaded)
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


class TextAgent(BaseAgent):
    def _load_model(self):
        self._model, self._tokenizer = _model_cache.load_text_model(self.model_id)

    def _run(self, chunk: RawChunk, retry: bool = False, trace: _TraceHandle | None = None) -> ProcessedChunk:
        content = str(chunk.raw_content) + (self.RETRY_SUFFIX if retry else "")
        messages = [
            {"role": "system", "content": _TEXT_SYSTEM},
            {"role": "user", "content": f"PASSAGE:\n{content}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Token計測: apply_chat_template returns token IDs array
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
                # Estimate output tokens from response text
                output_tokens = len(raw.split())  # Rough estimate: ~1 token per word
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


class TableAgent(BaseAgent):
    def _load_model(self):
        self._model, self._tokenizer = _model_cache.load_text_model(self.model_id)

    def _run(self, chunk: RawChunk, retry: bool = False, trace: _TraceHandle | None = None) -> ProcessedChunk:
        content = str(chunk.raw_content) + (self.RETRY_SUFFIX if retry else "")
        messages = [
            {"role": "system", "content": _TABLE_SYSTEM},
            {"role": "user", "content": f"TABLE:\n{content}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Token計測: apply_chat_template returns token IDs array
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
                output_tokens = len(raw.split())  # Rough estimate: ~1 token per word
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


class VisionAgent(BaseAgent):
    def _load_model(self):
        try:
            self._model, self._processor, self._config = _model_cache.load_vision_model(self.model_id)
            self._use_vision = True
        except Exception as e:
            log.warning("VisionAgent: vision model failed (%s). OCR fallback.", e)
            self._use_vision = False

    def _run(self, chunk: RawChunk, retry: bool = False, trace: _TraceHandle | None = None) -> ProcessedChunk:
        if not self._use_vision:
            return self._ocr_fallback(chunk)
        img = chunk.raw_content
        extra = self.RETRY_SUFFIX if retry else ""
        user_text = f"Describe.{extra}"

        # Combine system message with user text
        full_prompt = f"{_VISION_SYSTEM}\n\n{user_text}"
        prompt = apply_chat_template(self._processor, self._config, full_prompt, num_images=1)

        # Generation with tracing (VLM - token count not available)
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

    def _ocr_fallback(self, chunk: RawChunk) -> ProcessedChunk:
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


# ═══════════════════════════════════════════════════════════
# 7. AGENT ROUTER
# ═══════════════════════════════════════════════════════════


class AgentRouter:
    def __init__(self, text: TextAgent, table: TableAgent, vision: VisionAgent):
        self._map = {ChunkType.TEXT: text, ChunkType.TABLE: table, ChunkType.FIGURE: vision}

    def route(self, chunk: RawChunk, trace: _TraceHandle | None = None) -> ProcessedChunk:
        return self._map[chunk.chunk_type].process(chunk, trace=trace)


# ═══════════════════════════════════════════════════════════
# 8. CHUNK STORE  (embedding model — always loaded, lightweight)
# ═══════════════════════════════════════════════════════════


class ChunkStore:
    EMBED_MODEL = "intfloat/multilingual-e5-small"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._embedder = SentenceTransformer(self.EMBED_MODEL)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection("agentic_rag", metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[ProcessedChunk]) -> None:
        texts = [f"{c.structured_text}\n\n{c.intuition_summary}" for c in chunks]
        embs = self._embedder.encode(texts, normalize_embeddings=True).tolist()
        metadatas = []
        for c in chunks:
            m = {
                "chunk_type": c.chunk_type.value,
                "page_num": c.page_num,
                "source_file": c.source_file,
                "intuition_summary": c.intuition_summary,
                "key_concepts": json.dumps(c.key_concepts, ensure_ascii=False),
                "confidence": c.confidence,
                "agent_notes": c.agent_notes,
            }
            if c.validation is not None:
                m["validation_score"] = c.validation.verdict_score
                m["validation_issues"] = "; ".join(c.validation.issues)
            metadatas.append(m)
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embs,
            documents=[c.structured_text for c in chunks],
            metadatas=metadatas,
        )
        log.info("Upserted %d chunks.", len(chunks))

    def query(self, question: str, n_results: int = 6, chunk_type: ChunkType | None = None) -> list[dict]:
        vec = self._embedder.encode([question], normalize_embeddings=True).tolist()
        where = {"chunk_type": chunk_type.value} if chunk_type else None
        res = self._col.query(
            query_embeddings=vec, n_results=n_results, where=where, include=["documents", "metadatas", "distances"]
        )
        return [
            {"text": doc, "meta": meta, "score": 1 - dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


# ═══════════════════════════════════════════════════════════
# 9. REASONING ORCHESTRATOR AGENT  (BaseLoadableModel — 10B)
#    Now explicitly loadable/unloadable.
#    Retrieval (ChunkStore.query) does NOT require this model to be loaded.
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


class ReasoningOrchestratorAgent(BaseLoadableModel):
    """
    10B reasoning SLM orchestrator — now a BaseLoadableModel.

    Split into:
        retrieve(question, store, trace)  → list[dict]   (no model needed)
        generate(question, hits, trace)   → RAGAnswer    (model required)

    This split lets the pipeline:
      1. Retrieve first (no model loaded → no VRAM)
      2. Load orchestrator
      3. Generate
      4. Unload orchestrator  ← VRAM freed before answer validator loads
    """

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

    def _do_load(self) -> None:
        self._model, self._tokenizer = _model_cache.load_text_model(self.model_id)

    def _do_unload(self) -> None:
        del self._model
        del self._tokenizer

    # ── Retrieval (no model needed) ────────────────────────

    def retrieve(
        self,
        question: str,
        store: ChunkStore,
        trace: _TraceHandle | None = None,
    ) -> list[dict]:
        def _do():
            hits = store.query(question, n_results=8)
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
        trace: _TraceHandle | None = None,
    ) -> RAGAnswer:
        self._assert_loaded()
        context_str = self._build_context(hits)
        prompt = _ORCHESTRATOR_SYSTEM.format(context=context_str, question=question)
        messages = [{"role": "user", "content": prompt}]

        formatted_prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Token計測: apply_chat_template returns token IDs array
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
                output_tokens = len(output.split())  # Rough estimate: ~1 token per word
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
        match = self._THINK_RE.search(raw)
        if match:
            return match.group(1).strip(), self._THINK_RE.sub("", raw).strip()
        return "", raw.strip()

    @staticmethod
    def _build_context(hits: list[dict]) -> str:
        parts = []
        for i, h in enumerate(hits, 1):
            m = h["meta"]
            parts.append(
                f"[{i}] ({m['chunk_type'].upper()} | {m['source_file']} p.{m['page_num']} | "
                f"score={h['score']:.2f})\n"
                f"Summary: {m['intuition_summary']}\nContent: {h['text'][:800]}"
            )
        return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# 10. PIPELINE v3  — sequential load/unload for all 10B+ models
# ═══════════════════════════════════════════════════════════


class AgenticRAGPipeline:
    """
    v3: Sequential load/unload for heavy models.

    Model memory timeline:

    ingest():
      Phase 1 — Extraction  (small SLMs, always loaded):
        TextAgent(3-4B) + TableAgent(3B) + VisionAgent(2B) run concurrently in VRAM.
        Total: ~8-9 GB. Acceptable for 16 GB cards.

      Phase 2 — Chunk Validation  (ChunkValidatorAgent, Qwen2-VL-7B):
        [LOAD]   ChunkValidatorAgent  (+14 GB)
        run validate_chunk() for all chunks
        [UNLOAD] ChunkValidatorAgent  (-14 GB + CUDA cache clear)

    query():
      Phase 1 — Retrieval  (embedding model only, ~120 MB):
        retrieve() — no LLM needed

      Phase 2 — Generation  (OrchestratorAgent ~16 GB):
        [LOAD]   OrchestratorAgent
        generate()
        [UNLOAD] OrchestratorAgent  (-16 GB + CUDA cache clear)

      Phase 3 — Answer Validation  (AnswerValidatorAgent ~16 GB):
        [LOAD]   AnswerValidatorAgent
        validate_answer()
        [UNLOAD] AnswerValidatorAgent  (-16 GB + CUDA cache clear)

    Peak VRAM requirement: max(small_SLMs + chunk_validator, orchestrator, answer_validator)
      ≈ max(~22 GB, ~16 GB, ~16 GB)
      → 24 GB GPU (e.g. RTX 4090, A10G) sufficient with 4-bit quant on small SLMs.
      → 16 GB GPU: enable lazy_agents=True (small SLMs also load/unload per chunk).
    """

    @classmethod
    def build(
        cls,
        text_model: str = "mlx-community/Phi-3.5-mini-Instruct-4bit",
        table_model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
        vision_model: str = "mlx-community/SmolVLM-256M-Instruct-4bit",
        orchestrator_model: str = "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        chunk_validator_model: str = "mlx-community/Qwen2-VL-7B-Instruct-4bit",  # ← Checkpoint A
        answer_validator_model: str = "mlx-community/Qwen3-8B-4bit",  # ← Checkpoint B
        persist_dir: str = "./chroma_db",
        lazy_agents: bool = False,  # True → small SLMs also load/unload per chunk
    ) -> "AgenticRAGPipeline":
        log.info("=" * 70)
        log.info("🚀 Initializing Agentic RAG Pipeline")
        log.info("=" * 70)

        obj = cls()
        obj.lazy_agents = lazy_agents

        log.info("📂 Setting up vector store: %s", persist_dir)
        obj.parser = PDFParser()
        obj.store = ChunkStore(persist_dir)
        log.info("✓ Vector store initialized")

        log.info("📡 Setting up Langfuse tracer...")
        obj.tracer = LangfuseTracer()
        log.info("✓ Tracer initialized")

        # Small SLMs — load immediately (stay loaded throughout ingest)
        log.info("\n📦 Loading extraction agents (small SLMs - stay loaded):")

        log.info("  🚀 Text agent: %s", text_model)
        text_agent = TextAgent(text_model)
        log.info("  ✓ Text agent ready")

        log.info("  🚀 Table agent: %s", table_model)
        table_agent = TableAgent(table_model)
        log.info("  ✓ Table agent ready")

        log.info("  🚀 Vision agent: %s", vision_model)
        vision_agent = VisionAgent(vision_model)
        log.info("  ✓ Vision agent ready")

        obj.router = AgentRouter(text_agent, table_agent, vision_agent)
        log.info("\n✅ Extraction agents loaded")

        # Heavy models — instantiate WITHOUT loading; load/unload per phase
        log.info("\n📋 Initializing validator agents (lazy-loaded on demand):")

        log.info("  📋 Orchestrator: %s", orchestrator_model)
        obj.orchestrator = ReasoningOrchestratorAgent(orchestrator_model)
        log.info("  ✓ Orchestrator initialized")

        log.info("  📋 Chunk validator: %s", chunk_validator_model)
        obj.chunk_validator = ChunkValidatorAgent(chunk_validator_model)
        log.info("  ✓ Chunk validator initialized")

        log.info("  📋 Answer validator: %s", answer_validator_model)
        obj.answer_validator = AnswerValidatorAgent(answer_validator_model)
        log.info("  ✓ Answer validator initialized")

        log.info("\n" + "=" * 70)
        log.info("✅ Pipeline ready for ingestion and querying")
        log.info("=" * 70 + "\n")

        return obj

    # ── Ingestion ──────────────────────────────────────────

    def ingest(
        self,
        pdf_path: str | Path,
        validates: bool = True,
    ) -> list[ProcessedChunk]:
        pdf_path = Path(pdf_path)

        log.info("=" * 70)
        log.info("📂 INGEST PHASE: %s", pdf_path.name)
        log.info("=" * 70)

        with self.tracer.trace(
            "ingest_pdf",
            input={"file": pdf_path.name, "validates": validates},
            metadata={"pipeline": "agentic_rag_v3"},
        ) as trace:
            # ── Phase 1: Parse ─────────────────────────────
            log.info("📄 Parsing PDF...")
            with trace.span("parse_pdf") as s:
                raw_chunks = self.parser.parse(pdf_path)
                log.info("✓ Parsed %d raw chunks (text/table/figure)", len(raw_chunks))
                s.update(output={"n_raw": len(raw_chunks)})

            # ── Phase 2: Extract (small SLMs always loaded) ─
            log.info("🔄 Extracting chunks with agents...")
            extracted: list[tuple[RawChunk, ProcessedChunk]] = []
            for raw in raw_chunks:
                processed = self.router.route(raw, trace=trace)
                extracted.append((raw, processed))

            # ── Phase 3: Chunk Validation (load → run → unload) ─
            accepted: list[ProcessedChunk] = []
            corrected_count = 0
            discarded_count = 0

            if validates:
                log.info("✅ CHECKPOINT A: Starting chunk validation...")
                with self.chunk_validator:  # ← load on enter, unload on exit
                    log.info("  [LOAD] ChunkValidatorAgent loaded")
                    for idx, (raw, processed) in enumerate(extracted, 1):
                        val = self.chunk_validator.validate_chunk(raw=raw, processed=processed, trace=trace)
                        processed.validation = val

                        self.tracer.score(
                            trace_id=trace.trace_id,
                            name="chunk_quality",
                            value=val.verdict_score,
                            comment=f"p.{processed.page_num} {processed.chunk_type.value} | " + "; ".join(val.issues),
                        )

                        if not val.is_valid:
                            if val.corrected is not None:
                                val.corrected.validation = val
                                accepted.append(val.corrected)
                                corrected_count += 1
                                log.debug(
                                    "  ↻ p.%d %s — corrected by validator",
                                    processed.page_num,
                                    processed.chunk_type.value,
                                )
                            else:
                                discarded_count += 1
                                log.debug(
                                    "  ✗ p.%d %s — discarded",
                                    processed.page_num,
                                    processed.chunk_type.value,
                                )
                        elif processed.confidence >= 0.25:
                            accepted.append(processed)
                        else:
                            discarded_count += 1
                # ← ChunkValidatorAgent.unload() called here automatically
                log.info("  [UNLOAD] ChunkValidatorAgent unloaded")
                log.info("✓ Chunk validation complete: %d corrected, %d discarded", corrected_count, discarded_count)

            else:
                # Skip validation — accept all chunks above confidence floor
                accepted = [p for (_, p) in extracted if p.confidence >= 0.25]
                log.info("⊘ Validation skipped")

            log.info(
                "Ingestion result: accepted=%d corrected=%d discarded=%d",
                len(accepted),
                corrected_count,
                discarded_count,
            )

            # ── Phase 4: Upsert ────────────────────────────
            log.info("💾 Upserting %d chunks into vector store...", len(accepted))
            with trace.span("upsert_store", input={"n": len(accepted)}) as s:
                self.store.upsert(accepted)
                s.update(output={"upserted": len(accepted)})
            log.info("✓ Chunks stored")

            log.info("=" * 70 + "\n")

        return accepted

    # ── Query ──────────────────────────────────────────────

    def query(
        self,
        question: str,
        session_id: str | None = None,
        validates: bool = True,
    ) -> RAGAnswer:
        log.info("=" * 70)
        log.info("🔍 QUERY PHASE: %s", question[:80])
        log.info("=" * 70)

        with self.tracer.trace(
            "rag_query",
            input={"question": question, "validates": validates},
            session_id=session_id,
        ) as trace:
            # ── Phase 1: Retrieve (embedding model only, no LLM) ─
            log.info("📚 Retrieving relevant chunks...")
            hits = self.orchestrator.retrieve(question, self.store, trace=trace)
            log.info("✓ Retrieved %d chunks", len(hits))

            # ── Phase 2: Generate (load orchestrator → generate → unload) ─
            log.info("🤖 Generating answer with orchestrator...")
            with self.orchestrator:  # ← load on enter, unload on exit
                log.info("  [LOAD] OrchestratorAgent loaded")
                result = self.orchestrator.generate(question, hits, trace=trace)
                log.info("  ✓ Answer generated")
            # ← OrchestratorAgent.unload() called here — VRAM freed
            log.info("  [UNLOAD] OrchestratorAgent unloaded")

            result.trace_id = trace.trace_id

            if validates:
                # ── Phase 3: Validate answer (load answer_validator → validate → unload) ─
                log.info("✅ CHECKPOINT B: Starting answer validation...")
                source_texts = [sc["text"] for sc in result.source_chunks]

                with self.answer_validator:  # ← load on enter, unload on exit
                    log.info("  [LOAD] AnswerValidatorAgent loaded")
                    ans_val = self.answer_validator.validate_answer(
                        question=question,
                        answer=result,
                        source_texts=source_texts,
                        trace=trace,
                    )
                    log.info("  ✓ Answer validation complete")
                # ← AnswerValidatorAgent.unload() called here — VRAM freed
                log.info("  [UNLOAD] AnswerValidatorAgent unloaded")

                self.tracer.score(
                    trace_id=trace.trace_id,
                    name="answer_grounding",
                    value=ans_val.verdict_score,
                    comment=f"grounded={ans_val.is_grounded} | " + "; ".join(ans_val.hallucinations),
                )

                was_revised = False
                if not ans_val.is_grounded:
                    if ans_val.revised_answer:
                        log.warning(
                            "⚠️  Hallucinations detected — substituting revised answer\n  Hallucinations: %s",
                            ans_val.hallucinations,
                        )
                        result.answer = ans_val.revised_answer
                        was_revised = True
                    else:
                        log.warning(
                            "⚠️  Hallucinations detected, no revision available\n  Hallucinations: %s",
                            ans_val.hallucinations,
                        )
                        result.answer = "[VALIDATION WARNING: claims may not be grounded]\n\n" + result.answer

                result.validation_summary = ValidationSummary(
                    answer_is_grounded=ans_val.is_grounded,
                    hallucinations=ans_val.hallucinations,
                    answer_verdict_score=ans_val.verdict_score,
                    validator_notes=ans_val.validator_notes,
                    answer_was_revised=was_revised,
                )
            else:
                log.info("⊘ Answer validation skipped")

        log.info("=" * 70)
        log.info("✅ Query complete - Trace ID: %s", result.trace_id)
        log.info("=" * 70 + "\n")

        return result


# ═══════════════════════════════════════════════════════════
# 11. DEMO
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys

    # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
    # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

    if len(sys.argv) < 2:
        print("Usage: python agentic_rag_flow.py <pdf_path> [question]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Summarise the main findings. Describe any key figures or tables."

    rag = AgenticRAGPipeline.build(
        orchestrator_model="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        chunk_validator_model="mlx-community/SmolVLM-256M-Instruct-4bit",
        answer_validator_model="mlx-community/Qwen3-8B-4bit",
    )

    print(f"\n[INGEST] {pdf_path}")
    chunks = rag.ingest(pdf_path, validates=True)
    stats = {ct.value: sum(1 for c in chunks if c.chunk_type == ct) for ct in ChunkType}
    print(f"[CHUNK STATS] {stats}")

    # Save chunks to output directory
    save_chunks(chunks, pdf_path)

    print(f"\n[QUERY] {question}")
    result = rag.query(question, validates=True)

    print("\n=== ANSWER ===")
    print(result.answer)

    if result.validation_summary:
        v = result.validation_summary
        print("\n=== VALIDATION SUMMARY ===")
        print(f"  Grounded       : {v.answer_is_grounded}")
        print(f"  Verdict score  : {v.answer_verdict_score:.2f}")
        print(f"  Was revised    : {v.answer_was_revised}")
        if v.hallucinations:
            print(f"  Hallucinations : {v.hallucinations}")

    print(f"\n[Langfuse trace ID] {result.trace_id}")

    # Save answer to output directory
    save_answer(result, pdf_path, question)

    # Clean up unused models from cache
    log.info("🧹 Cleaning up unused models...")
    _model_cache.cleanup_unused_models()
