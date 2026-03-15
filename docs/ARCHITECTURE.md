# Agentic RAG Architecture

Version: 6.x（implementation-aligned）
Last updated: 2026-03-15

この文書は、現行コードベースの実装事実に合わせたアーキテクチャ仕様です。

## 1. システム概要

本システムは、PDFをチャンク化してベクトルストアに保存し、質問に対して根拠付き回答を返すRAGパイプラインです。

主要レイヤー:
- Parse layer: PDFをRawChunkへ変換
- Extraction layer: チャンク種別ごとの抽出
- Validation layer: チャンク品質/回答根拠性の検証
- Retrieval layer: ChromaDB+埋め込み検索
- Generation layer: 推論モデルで回答生成
- Serialization/Audit layer: JSON/HTML出力

## 2. ディレクトリ別責務

### 2.1 Core（`src/core`）

- `models.py`
  - ドメインデータモデル定義
  - `RawChunk`, `ProcessedChunk`, `RAGAnswer`
  - `ChunkValidationResult`, `AnswerValidationResult`, `CrossLinkMetadata`
- `parser.py`
  - PDF解析（`pymupdf`, `pdfplumber`）
  - テキスト/表/図のraw chunk生成
- `store.py`
  - ChromaDBラッパー
- `pipeline.py`
  - 標準Sequentialパイプライン（ingest/query）
- `langgraph_pipeline.py`
  - LangGraph ingest/queryパイプライン
- `graph_state.py`
  - LangGraph状態スキーマ（`QueryState`, `IngestState`）
- `crewai_pipeline.py`
  - CrewAI統合経路
- `cache.py`
  - モデルキャッシュ管理
- `config.py`
  - `settings.json`読み込み

### 2.2 Agents（`src/agents`）

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

### 2.3 Integrations（`src/integrations`）

- `langfuse.py`
  - trace/span/score送信
- `dspy_adapter.py`, `dspy_modules.py`
  - DSPyとMLXの接続
- `crew_mlx_tools.py`
  - CrewAI toolインターフェイス

### 2.4 Utils（`src/utils`）

- `serialization.py`
  - chunks/answer JSON保存
- `audit.py`
  - 監査用JSON/HTML/画像出力

## 3. 実行モード

### 3.1 Sequential（標準）

- 実装: `src/core/pipeline.py`
- 対応: ingest/query/pipeline
- 特徴:
  - 単純で読みやすい経路
  - 挙動の追跡・デバッグがしやすい

### 3.2 LangGraph

- 実装: `src/core/langgraph_pipeline.py`
- 対応:
  - ingest: `LangGraphIngestPipeline`
  - query: `LangGraphQueryPipeline`
  - pipeline: queryフェーズで適用
- 特徴:
  - ノード単位で状態遷移を明示
  - 条件分岐とフェイルセーフを組み込みやすい

### 3.3 CrewAI

- 実装: `src/core/crewai_pipeline.py`, `src/integrations/crew_mlx_tools.py`
- 対応: ingest/query/pipeline（`--use-crewai`）
- 実装方針:
  - 外部API依存を避けるため、ローカルMLX処理を優先
  - 抽出フェーズは`AgentRouter`経由のローカル抽出を使用（Extraction crew task実行はスキップ）
  - `--no-validate`指定時はingest validationフェーズをスキップ
  - 一部Crewは段階的実装/簡略実装

## 4. データフロー

### 4.1 Ingest

- 入力PDF
- `PDFParser.parse()`で`RawChunk`群を生成
- `AgentRouter`がchunk typeに応じて抽出
- （任意）CHECKPOINT Aで品質検証
- `ChunkStore.upsert()`で永続化
- （任意）`save_chunk_audit()`で監査ファイル生成

### 4.2 Query

- 質問入力
- `ChunkStore`から関連チャンク取得
- `ReasoningOrchestratorAgent.generate()`で回答生成
- （任意）CHECKPOINT Bで根拠性検証
- `RAGAnswer`を返却
- （任意）`save_answer()`でJSON保存

## 5. Parserのテーブル検出方針

現行のparserはprecision-firstポリシーです。

- 標準候補優先:
  - `pdfplumber.find_tables()`の標準候補を優先
- fallback候補の適用条件:
  - 標準候補が0件のページでは常にtext strategy fallbackを実行
  - `enable_figure_aware_fallback=True`かつページにfigureがある場合もfallbackを実行
- 誤検出抑制:
  - figure重なり
  - caption cue（table/figure）
  - 数値セル率
  - 長文セル率（prose-like判定）

この方針は本文のtable誤検出を抑える一方で、borderless tableの一部取りこぼしが発生し得ます。図中表はVision/Table extraction経路で補完します。

## 6. 状態モデル

### 6.1 QueryState（`src/core/graph_state.py`）

主要フィールド:
- 入力: `question`, `validates`, `session_id`
- 中間結果: `retrieved_hits`, `raw_answer`, `validated_answer`, `final_answer`
- メタ: `trace`, `errors`, `warnings`, `stats`
- 制御: `current_phase`, `needs_revision`, `skip_validation`, `insufficient_context`

### 6.2 IngestState（`src/core/graph_state.py`）

主要フィールド:
- 入力: `pdf_path`, `validates`, `storage_dir`
- 中間結果: `raw_chunks`, `all_extracted`, `accepted_chunks`
- 内部キー: `_extracted_pairs`, `_audit_output_dir`
- メタ/制御: `trace`, `errors`, `warnings`, `stats`, `current_phase`

## 7. バリデーション設計

### 7.1 CHECKPOINT A（Chunk Validation）

- 対象: 抽出済みチャンク
- 目的: 保存前の品質担保
- 出力: `ChunkValidationResult`

### 7.2 CHECKPOINT B（Answer Validation）

- 対象: 生成済み回答
- 目的: 根拠性確認、幻覚検出
- 出力: `AnswerValidationResult`
- DSPy統合: AnswerValidator側で利用

## 8. モデル管理とメモリ戦略

### 8.1 BaseLoadableModel

`BaseLoadableModel`は重いモデルを必要時にロード/アンロードするための共通インターフェイスです。

- `load()`
- `unload()`
- `with model:`パターンで自動管理

### 8.2 キャッシュ

- `src/core/cache.py`でモデルキャッシュを管理
- 目的:
  - 再ロードコスト削減
  - モデル再利用

## 9. 出力と監査

### 9.1 シリアライズ

`src/utils/serialization.py`の`OUTPUT_DIR=./output`へ保存されます。

- `save_chunks()`
  - `<pdf_stem>_chunks.json`
- `save_answer()`
  - `<pdf_stem>_answer.json`
  - queryコマンドでは`query_answer.json`

### 9.2 監査

`save_chunk_audit()`は以下を生成します。

- `<pdf_stem>_audit.json`
- `<pdf_stem>_audit.html`
- `<pdf_stem>_audit/pages/*.png`
- `<pdf_stem>_audit/figures/*.png`

`--output`は主に監査出力先として利用されます。

## 10. 観測性

- `LangfuseTracer`経由でtrace/span/scoreを送信
- 環境変数未設定時はno-opとして動作継続

## 11. 既知の注意点

- MLX依存のためApple Silicon前提
- OCRは自動起動しない（スキャンPDFは前処理が必要）
- `pipeline --use-langgraph`はqueryフェーズに適用
- parserはprecision-firstなので、誤検出抑制と取りこぼしのトレードオフがある
