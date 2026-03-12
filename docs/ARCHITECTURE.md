# Agentic RAG Architecture

Version: 5.x (implementation-aligned)
Last updated: 2026-03-12

この文書は、現行コードベースの実装事実に合わせて構成したアーキテクチャ仕様です。

## 1. システム概要

本システムは、PDFをチャンク化してベクトルストアに保存し、質問に対して根拠付き回答を返すRAGパイプラインです。

主要な処理レイヤー:

- Parse layer: PDFをRawChunkへ変換
- Extraction layer: チャンク種別ごとの抽出
- Validation layer: チャンク品質/回答根拠性の検証
- Retrieval layer: ChromaDB + 埋め込み検索
- Generation layer: 推論モデルで回答生成
- Serialization/Audit layer: JSON/HTML出力

## 2. ディレクトリ別責務

### 2.1 Core (`src/core`)

- `models.py`
- ドメインデータモデル定義
- `RawChunk`, `ProcessedChunk`, `RAGAnswer`
- `ChunkValidationResult`, `AnswerValidationResult`, `CrossLinkMetadata`
- `parser.py`
- PDF解析 (`pymupdf`, `pdfplumber`)
- `store.py`
- ChromaDBラッパー
- `pipeline.py`
- 標準Sequentialパイプライン
- `langgraph_pipeline.py`
- Query用LangGraphパイプライン
- `graph_state.py`
- LangGraph状態スキーマ (`QueryState`, `IngestState`)
- `crewai_pipeline.py`
- CrewAI関連の取り込み処理
- `cache.py`
- モデルキャッシュ管理
- `config.py`
- `settings.json` 読み込み

### 2.2 Agents (`src/agents`)

- `base.py`
- `BaseAgent`, `BaseLoadableModel`
- `extraction.py`
- `TextAgent`, `TableAgent`, `VisionAgent`
- `validation.py`
- `ChunkValidatorAgent`, `AnswerValidatorAgent`
- `orchestrator.py`
- `ReasoningOrchestratorAgent`
- `router.py`
- チャンクタイプ別ルーティング
- `crewai_agents.py`
- CrewAI向けエージェント定義

### 2.3 Integrations (`src/integrations`)

- `langfuse.py`
- トレース/スパン/スコア送信
- `dspy_adapter.py`, `dspy_modules.py`
- DSPyとMLXの接続
- `crew_mlx_tools.py`
- CrewAI toolインターフェイス

### 2.4 Utils (`src/utils`)

- `serialization.py`
- chunks/answerのJSON保存
- `audit.py`
- 監査用JSON/HTML/画像出力

## 3. 実行モード

### 3.1 Sequential (標準)

- 実装: `src/core/pipeline.py`
- 対応: ingest/query/pipeline
- 特徴:
- 実装がもっとも素直
- 検証フローを制御しやすい

### 3.2 LangGraph

- 実装: `src/core/langgraph_pipeline.py`
- 対応: query（およびpipelineのqueryフェーズ）
- ノード:
- retrieve
- check_quality
- generate_answer
- decide_validate
- validate_answer
- check_grounding
- revise_answer
- finalize
- 状態:
- `QueryState` を使用

### 3.3 CrewAI

- 実装: `src/core/crewai_pipeline.py`, `src/integrations/crew_mlx_tools.py`
- 対応: ingest/query/pipeline (`--use-crewai`)
- 実装方針:
- 外部API依存を避けるため、ローカルMLX処理を優先
- 一部Crewは段階的実装/簡略実装

## 4. データフロー

### 4.1 Ingest

- 入力PDF
- `PDFParser.parse()` で `RawChunk` 群を生成
- `AgentRouter` がchunk typeに応じて抽出
- （任意）CHECKPOINT Aで品質検証
- `ChunkStore.upsert()` で永続化
- （任意）`save_chunk_audit()`で監査ファイル生成

### 4.2 Query

- 質問入力
- `ChunkStore` から関連チャンク取得
- `ReasoningOrchestratorAgent.generate()` で回答生成
- （任意）CHECKPOINT Bで根拠性検証
- `RAGAnswer` を返却
- （任意）`save_answer()`でJSON保存

## 5. 状態モデル

### 5.1 QueryState (`src/core/graph_state.py`)

主要フィールド:

- 入力:
- `question`, `validates`, `session_id`
- 中間結果:
- `retrieved_hits`, `raw_answer`, `validated_answer`, `final_answer`
- メタ:
- `trace`, `errors`, `warnings`, `stats`
- 制御:
- `current_phase`, `needs_revision`, `skip_validation`, `insufficient_context`

### 5.2 IngestState

- LangGraphでのingest拡張用に定義済み
- 現時点で主経路はSequential ingest

## 6. バリデーション設計

### 6.1 CHECKPOINT A (Chunk Validation)

- 対象: 抽出済みチャンク
- 目的: 保存前の品質担保
- 出力: `ChunkValidationResult`

### 6.2 CHECKPOINT B (Answer Validation)

- 対象: 生成済み回答
- 目的: 根拠性確認、幻覚検出
- 出力: `AnswerValidationResult`
- DSPy統合: AnswerValidator側で利用

## 7. モデル管理とメモリ戦略

### 7.1 BaseLoadableModel

`BaseLoadableModel` は重いモデルを必要時にロード/アンロードするための共通インターフェイスです。

- `load()`
- `unload()`
- `with model:` パターンで自動管理

### 7.2 キャッシュ

- `src/core/cache.py` でモデルキャッシュを管理
- 目的:
- 再ロードコスト削減
- モデル再利用

## 8. 出力と監査

### 8.1 シリアライズ

- `save_chunks()`
- `<pdf_stem>_chunks.json`
- `save_answer()`
- `<pdf_stem>_answer.json`

### 8.2 監査

`save_chunk_audit()` は以下を生成します。

- `<pdf_stem>_audit.json`
- `<pdf_stem>_audit.html`
- `<pdf_stem>_audit/pages/*.png`
- `<pdf_stem>_audit/figures/*.png`

## 9. 観測性

- `LangfuseTracer` 経由でtrace/span/scoreを送信
- 環境変数未設定時はno-opとして動作継続

## 10. 実装済みとFutureの切り分け

実装済み:

- Sequential ingest/query
- LangGraph query
- CrewAI統合基盤
- DSPy (AnswerValidator)
- Langfuse tracing

Future/段階的拡張:

- LangGraph ingestの本格導入
- CrewAI側のフルオーケストレーション強化
- OCR自動前処理の標準統合

## 11. 既知の注意点

- `serialization.py` は `OUTPUT_DIR = ./output` を使用します。
- `--output` は主に監査レポート側の出力指定として使われます。

上記の出力仕様は、運用上の混乱を避けるためにREADMEのCLI仕様と合わせて確認することを推奨します。
