# Agentic RAG Architecture

Last updated: 2026-04-04

この文書は、現在サポートされている Sequential + Ollama 構成だけを説明します。削除済みの CrewAI、LangGraph、MLX 経路は対象外です。

## 1. System overview

システムは 2 フェーズです。

- ingest
  - PDF を parse して `RawChunk` を作る
  - 種別ごとの抽出を行い `ProcessedChunk` に変換する
  - 任意で CHECKPOINT A を実行する
  - ChromaDB に保存する
- query
  - ベクトルストアから関連チャンクを取得する
  - オーケストレータで回答を生成する
  - 任意で CHECKPOINT B を実行する

主なレイヤーは次の通りです。

- Parse layer
- Extraction layer
- Validation layer
- Retrieval layer
- Generation layer
- Serialization and audit layer

## 2. Directory responsibilities

### 2.1 `src/core`

- `config.py`
  - `settings.json` を読み込み、defaults と deep merge します。
- `models.py`
  - `RawChunk`, `ProcessedChunk`, `RAGAnswer` などのドメインモデルを定義します。
- `parser.py`
  - `pdfplumber` と `PyMuPDF` を使って PDF を解析し、text/table/figure の raw chunk を作ります。
- `pipeline.py`
  - 現在サポートされている sequential ingest/query runtime です。
- `store.py`
  - ChromaDB への upsert / retrieve を扱います。
- `cache.py`
  - Ollama client の取得と `HF_HOME` の設定を扱います。

### 2.2 `src/agents`

- `extraction.py`
  - `TextAgent`, `TableAgent`, `VisionAgent`
- `router.py`
  - `RawChunk.chunk_type` に応じて抽出 agent を選びます。
- `validation.py`
  - `ChunkValidatorAgent`, `AnswerValidatorAgent`
- `orchestrator.py`
  - `ReasoningOrchestratorAgent`

### 2.3 `src/integrations`

- `langfuse.py`
  - trace, span, score を送信します。設定がなければ no-op です。
- `dspy_adapter.py`, `dspy_modules.py`
  - DSPy と answer validation を接続します。

### 2.4 `src/utils`

- `serialization.py`
  - chunks / answer JSON を `./output` に保存します。
- `audit.py`
  - 監査用 JSON / HTML / 画像を出力します。

## 3. Runtime flow

### 3.1 Ingest

`AgenticRAGPipeline.ingest()` は次の順で進みます。

1. `PDFParser.parse()`
2. `AgentRouter.route_with_policy()` による抽出
3. 任意の chunk validation
4. `ChunkStore.upsert()`
5. 任意の監査出力

### 3.2 Query

`AgenticRAGPipeline.query()` は次の順で進みます。

1. retrieve
2. orchestrator generate
3. 任意の answer validation
4. 必要なら revised answer へ置換

## 4. Parser behavior

### 4.1 Table detection

`PDFParser` は precision-first です。

- まず `pdfplumber.find_tables()` の標準候補を使います。
- fallback の text strategy は次で有効になります。
  - 標準候補が 0 件
  - `enable_figure_aware_fallback=True` かつ figure があるページ
- 誤検出抑制のために figure overlap、caption cue、numeric ratio、long-cell ratio を見ます。

### 4.2 OCR

OCR 設定は `src/core/config.py` 経由で読み込まれます。

- `ocr.engine`
- `ocr.default_lang`
- `ocr.japanese_lang`
- `ocr.fallback_lang`
- `ocr.config`

既定の OCR engine は `easyocr` です。実運用の注意点は [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md) を参照してください。

## 5. Validation

### 5.1 CHECKPOINT A

- 対象: 抽出済み chunk
- 実装: `ChunkValidatorAgent`
- 目的: 保存前の品質担保
- 挙動:
  - valid なら採用
  - corrected が返れば差し替えて採用
  - invalid かつ corrected なしは破棄

### 5.2 CHECKPOINT B

- 対象: 生成済み answer
- 実装: `AnswerValidatorAgent`
- 目的: grounding の確認と hallucination 検出
- 挙動:
  - grounded ならそのまま返す
  - revised answer があれば差し替える
  - revised answer がなければ警告を残したまま返す

DSPy 連携は `AnswerValidatorAgent` で使われます。

## 6. Model and memory handling

- 抽出 agent は build 時に初期化されます。
- orchestrator, chunk validator, answer validator は必要フェーズだけ `with model:` で load / unload されます。
- LLM の実メモリ管理は Ollama server が担当します。
- `src/core/cache.py` は sentence-transformers 用に `HF_HOME` を `~/.models` に設定します。

## 7. Outputs and audit

### 7.1 JSON outputs

`src/utils/serialization.py` は次を `./output` に保存します。

- `<pdf_stem>_chunks.json`
- `<pdf_stem>_answer.json`

### 7.2 Audit outputs

`save_chunk_audit()` は次を出力します。

- `<pdf_stem>_audit.json`
- `<pdf_stem>_audit.html`
- `pages/*.png`
- `figures/*.png`

監査出力先は CLI の `--output` から渡されます。

## 8. Observability

`LangfuseTracer` が trace / span / score を記録します。環境変数が未設定でも処理は継続します。

- ingest trace
- query trace
- chunk quality score
- answer grounding score

`session_id` は関連 query を Langfuse 上で束ねるために使います。

## 9. Implementation notes

- `app.py` の CLI 既定モデル値は `settings.json` より優先されます。
- `lazy_agents` は保持されますが、現時点では大きな実行分岐には接続されていません。
- サポートされる実行モードは Sequential のみです。
