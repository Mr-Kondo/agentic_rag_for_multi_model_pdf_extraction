# Agentic RAG Flow — Architecture Notes

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PDF INPUT                                    │
│            (academic paper / government publication)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ PDFParser   │  pymupdf + pdfplumber
                    │             │  → RawChunk(TEXT | TABLE | FIGURE)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │      AgentRouter        │
              │  (dispatches by type)   │
              └──┬──────────┬───────┬──┘
                 │          │       │
         ┌───────▼─┐  ┌─────▼──┐  ┌▼──────────┐
         │ Agent-1  │  │Agent-2 │  │  Agent-3  │
         │  TEXT    │  │ TABLE  │  │  VISION   │
         │  SLM     │  │  SLM   │  │   SLM     │
         │Phi-3.5   │  │Qwen2.5 │  │SmolVLM    │
         │ mini     │  │  3B    │  │  2B       │
         └────┬─────┘  └───┬────┘  └────┬──────┘
              │             │            │
              │   Self-Reflection Loop   │
              │   (retry if conf < 0.5)  │
              │             │            │
         ┌────▼─────────────▼────────────▼──────┐
         │         ProcessedChunk               │
         │  structured_text | intuition_summary │
         │  key_concepts | confidence           │
         └────────────────┬──────────────────────┘
                          │
                  ┌───────▼────────┐
                  │  ChunkStore    │
                  │  ChromaDB      │
                  │  e5-small      │
                  │  (multilingual)│
                  └───────┬────────┘
                          │
               ┌──────────▼──────────┐
               │  OrchestratorAgent  │
               │  - query routing    │
               │  - visual boost     │
               │  - provenance cite  │
               └──────────┬──────────┘
                          │
                    ┌─────▼──────┐
                    │   ANSWER   │
                    │ + sources  │
                    └────────────┘
```

## Agent Design Rationale

| Agent | Input | SLM Responsibility | Key Output Fields |
|-------|-------|-------------------|-------------------|
| TextAgent   | Raw prose text   | De-hyphenation, section normalisation, concept extraction | `structured_text`, `key_concepts` |
| TableAgent  | Markdown table   | Schema inference, unit extraction, merged-cell repair    | `structured_text`, `schema` |
| VisionAgent | PIL.Image        | Figure type classification, axis reading, flow description | `figure_type`, `intuition_summary` |

## Self-Reflection Mechanism

Each agent checks its own `confidence` score (0–1) in the JSON response.
If `confidence < 0.5`, the agent re-runs with a stricter prompt addendum.
This is a lightweight substitute for a separate critique model.

## Retrieval Strategy

1. Standard semantic search (all chunk types) — top-8 results
2. If question contains visual keywords (graph, figure, フロー etc.) → additional top-3 FIGURE-only search, merged deduplicated
3. Context window to orchestrator: 8–11 chunks × 800 chars each ≈ ~6,400–8,800 tokens

## Model Swap Guide

| Slot          | Default              | Alternative (lighter)        | Alternative (heavier) |
|---------------|----------------------|------------------------------|-----------------------|
| Text SLM      | Phi-3.5-mini (3.8B)  | Qwen2.5-1.5B-Instruct        | Mistral-7B-Instruct   |
| Table SLM     | Qwen2.5-3B           | Same as Text SLM             | Phi-3.5-mini          |
| Vision SLM    | SmolVLM-2B           | Qwen2-VL-2B-Instruct         | Phi-3.5-vision (4.2B) |
| Orchestrator  | Phi-3.5-mini (3.8B)  | Any of the above             | Claude via API        |
| Embedder      | e5-small-multilingual| paraphrase-multilingual-mpnet| text-embedding-3-small|

## Critical Limitations

1. **Table detection accuracy**: pdfplumber struggles with borderless tables and merged cells in scanned PDFs. Camelot (lattice mode) is more robust for scanned documents but requires Ghostscript.
2. **Vision SLM hallucination**: 2B-class models reliably describe layout but may misread axis values or numerical data. Always cross-check with Table Agent when figure contains data grids.
3. **OCR is NOT invoked for text**: This pipeline assumes a digitally-native PDF. For scanned PDFs, prepend an OCR step (pytesseract, EasyOCR, or AWS Textract) before parsing.
4. **Confidence scores are self-assessed**: The SLM judges its own output — this is unreliable for adversarial or out-of-distribution inputs. An external scoring model would be more robust.
5. **Memory**: Loading 3 SLMs simultaneously requires ≥16 GB VRAM (GPU) or ≥32 GB RAM (CPU). Use 4-bit quantisation (bitsandbytes) or load/unload agents sequentially if constrained.

## Install

```bash
pip install unstructured[all-docs] pdfplumber pymupdf pillow
pip install transformers torch accelerate bitsandbytes
pip install chromadb sentence-transformers
pip install pytesseract  # optional fallback OCR
```

## Run

```bash
python agentic_rag_flow.py my_paper.pdf "What method does Figure 2 illustrate?"
```
