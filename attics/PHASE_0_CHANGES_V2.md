# v2 Changes: Reasoning Orchestrator + Langfuse

## What changed from v1

| Component | v1 | v2 |
|---|---|---|
| Orchestrator model | Phi-3.5-mini (3.8B, standard) | DeepSeek-R1-Distill-Llama-8B (8B, reasoning) |
| Orchestrator output | raw answer string | `RAGAnswer` dataclass with `answer` + `reasoning_trace` |
| Think-block handling | none | `_strip_reasoning()` separates CoT from answer |
| Observability | none | Langfuse — trace per ingest, trace per query |
| Trace injection | n/a | `trace: _TraceHandle` passed explicitly per call |
| Token usage | n/a | Logged to Langfuse generation span |
| Return type | `dict` | `RAGAnswer` (includes `trace_id`) |

---

## Langfuse Trace Hierarchy

```
Trace: ingest_pdf
  metadata: {file, pipeline}
  ├── Span: parse_pdf
  │     output: {n_raw: 47}
  ├── Span: agent_text      (×N)
  │     output: {confidence, key_concepts, notes}
  ├── Span: agent_table     (×M)
  │     output: {confidence, key_concepts, notes}
  ├── Span: agent_vision    (×K)
  │     output: {confidence, key_concepts, notes}
  └── Span: upsert_store
        output: {upserted: 44}

Trace: rag_query
  input: {question}
  session_id: (optional, for multi-turn)
  ├── Span: retrieve_chunks
  │     input: {question}
  │     output: {n_hits: 10}
  └── Generation: orchestrator_reasoning
        model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        input: {messages: [...]}
        output: <think>...</think> answer text
        usage: {input_tokens, output_tokens}
```

---

## Reasoning Model Options (~8-14B)

| Model | Params | VRAM (FP16) | CoT Format | Notes |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | ~16 GB | `<think>` | Best reasoning/size ratio |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | ~28 GB | `<think>` | Higher quality, heavier |
| `Qwen/Qwen3-8B` | 8B | ~16 GB | `<think>` | Good multilingual + Japanese |
| `Qwen/Qwen3-14B` | 14B | ~28 GB | `<think>` | Best Japanese understanding |

For 4-bit quantisation (halves VRAM requirement):
```python
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_4bit=True)
# Pass as: AutoModelForCausalLM.from_pretrained(..., quantization_config=quant_config)
```
The `pipeline()` API doesn't directly accept `quantization_config` — switch to
`AutoTokenizer` + `AutoModelForCausalLM` + manual `generate()` in `_generate()` if needed.

---

## Copilot Tuning Guide

### High-priority extension points (well-commented, designed for Copilot)

1. **`ReasoningOrchestratorAgent._retrieve()`**
   - Current: semantic + figure-boosted retrieval
   - Copilot target: add HyDE (hypothetical document embedding), BM25 hybrid, or MMR reranking

2. **`BaseAgent._run_with_retry()`**
   - Current: single retry on `confidence < 0.5`
   - Copilot target: add exponential backoff, circuit breaker, or alternative prompt strategy

3. **`ChunkStore.query()`**
   - Current: pure vector search
   - Copilot target: add metadata filter by `source_file`, `page_num` range, or `confidence` threshold

4. **`LangfuseTracer`**
   - Current: spans + generation
   - Copilot target: add `.score()` calls after each agent based on confidence, add dataset logging

5. **`PDFParser.parse()`**
   - Current: native PDF only
   - Copilot target: add EasyOCR preprocessing branch when `plumb_page.extract_text()` returns empty

### Naming conventions (consistent, Copilot-learnable)
- `_method()` = private implementation detail
- `method()` = public API
- `trace: _TraceHandle | None = None` = always optional, never required
- All agents share `process(chunk, trace)` signature
- All SLMs loaded in `_load_model()`, called in `_run(chunk, retry)`

---

## Environment Setup

```bash
# .env (load with python-dotenv or set in shell)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com     # or http://localhost:3000 for self-hosted

# Install
pip install langfuse
pip install transformers torch accelerate bitsandbytes
pip install chromadb sentence-transformers
pip install pdfplumber pymupdf pillow pytesseract
```
