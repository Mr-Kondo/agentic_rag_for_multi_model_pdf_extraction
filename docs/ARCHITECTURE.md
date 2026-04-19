# Agentic RAG Architecture

Last updated: 2026-04-19

この文書は、現行の Sequential + Ollama runtime の構成を説明します。
CrewAI / LangGraph / MLX 系の旧経路は対象外です。

## 1. System overview

システムは ingest と query の 2 フェーズです。

- ingest
  - PDF parse
  - chunk extraction (text/table/figure)
  - optional CHECKPOINT A
  - ChromaDB upsert
- query
  - retrieve
  - orchestrator generate
  - optional CHECKPOINT B

主要レイヤー:

- Parse
- Extraction
- Validation
- Retrieval
- Generation
- Serialization and audit

## 2. Directory responsibilities

### 2.1 src/core

- config.py: `_DEFAULTS` と `settings.json` の deep merge
- parser.py: PDF parse と OCR
- pipeline.py: ingest/query orchestration
- store.py: ChromaDB + BM25 ハイブリッド検索
- models.py: domain model
- cache.py: sentence-transformers 用 `HF_HOME` 設定

### 2.2 src/agents

- extraction.py: `TextAgent`, `TableAgent`, `VisionAgent`
- router.py: chunk type ベースの抽出ルーティング
- validation.py: `ChunkValidatorAgent`, `AnswerValidatorAgent`
- orchestrator.py: `ReasoningOrchestratorAgent`

### 2.3 src/integrations

- langfuse.py: trace/span/score
- dspy_adapter.py, dspy_modules.py: DSPy 接続

### 2.4 src/utils

- serialization.py: chunks/answer JSON 保存
- audit.py: audit JSON/HTML/image 保存
- token_counter.py: usage token 近似算出

## 3. Runtime flow

### 3.1 ingest

1. `PDFParser.parse()`
2. `AgentRouter.route_with_policy()`
3. optional CHECKPOINT A (`validates=True`)
4. `ChunkStore.upsert()`
5. optional audit export

### 3.2 query

1. retrieve
2. orchestrator generate
3. optional CHECKPOINT B (`validates=True`)
4. revised answer があれば差し替え

## 4. Parser behavior

### 4.1 Table detection

`PDFParser` は precision-first 方針です。

- `pdfplumber.find_tables()` の標準候補を優先
- fallback text strategy は次で有効
  - 標準候補 0 件
  - `enable_figure_aware_fallback=True` かつ figure ありページ
- 誤検出抑制に figure overlap / caption cue / numeric ratio / long-cell ratio を使用

### 4.2 OCR dispatch

OCR engine は region policy の `engine` に従って選択されます。

- `yomitoku`: `_ocr_text_from_bbox_yomitoku()`
- `easyocr`: `_ocr_text_from_bbox_easyocr()`
- `tesseract`: `_ocr_text_from_bbox_tesseract()`

`yomitoku` と `easyocr` は結果が空のとき `tesseract` にフォールバックします。

関連設定:

- `ocr.engine`
- `ocr.region_policies.*.engine`
- `ocr.yomitoku_device`
- `ocr.prewarm_yomitoku`
- `ocr.prewarm_easyocr`

詳細は [docs/OCR_ENGINE_BEHAVIOR.md](docs/OCR_ENGINE_BEHAVIOR.md) を参照してください。

## 5. Validation checkpoints

### 5.1 CHECKPOINT A

- ingest で実行
- `ChunkValidatorAgent`
- corrected が返れば置換、invalid かつ corrected なしは discard

### 5.2 CHECKPOINT B

- query で実行
- `AnswerValidatorAgent` + DSPy
- grounding 判定と必要時の answer revision

注: 実行制御は主に CLI の `--validate` / `--no-validate` です。

## 6. Output architecture

`--output` は chunks / answer / audit すべての保存先です。

- ingest
  - `<pdf_stem>_chunks.json`
  - `<pdf_stem>_audit.json`
  - `<pdf_stem>_audit.html`
  - `<pdf_stem>_audit/pages/*.png`
  - `<pdf_stem>_audit/figures/*.png`
- query
  - `query_answer.json` (保存時に `query.pdf` を stem として使うため)
- pipeline
  - `<pdf_stem>_chunks.json`
  - `<pdf_stem>_answer.json`
  - audit 一式

## 7. Observability

`LangfuseTracer` が ingest/query の trace を記録します。
環境変数未設定や SDK 非互換時は no-op で継続します。

- ingest trace
- query trace
- chunk quality score
- answer grounding score

## 8. Notes

- CLI のモデル default 表示は `config.get_model(...)` で解決されるため、`settings.json` があると値が変わります。
- `lazy_agents` は pipeline に保持されますが、現時点では実行分岐への寄与が限定的です。
- サポート runtime は Sequential のみです。
