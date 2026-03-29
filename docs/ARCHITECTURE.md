# Agentic RAG Architecture

Last updated: 2026-03-29

この文書は、現行コードの実装に合わせてシステム構成を説明するものです。将来構想ではなく、`app.py` と `src/` 以下で確認できる実装事実だけを扱います。

## 1. システム概要

システムは大きく2つのフェーズで構成されます。

- ingest
  - PDFをparseして`RawChunk`を作る
  - 種別ごとの抽出を行い `ProcessedChunk` に変換する
  - 任意でチャンク品質を検証する
  - ChromaDBに保存する
- query
  - ベクトルストアから関連チャンクを取得する
  - オーケストレータで回答を生成する
  - 任意で根拠性を検証する

主なレイヤーは次の通りです。

- Parse layer
- Extraction layer
- Validation layer
- Retrieval layer
- Generation layer
- Serialization / Audit layer

## 2. ディレクトリ責務

### 2.1 `src/core`

- `config.py`
  - `settings.json`を読み込み、deep mergeで不足キーを補完します。
- `models.py`
  - `RawChunk`, `ProcessedChunk`, `RAGAnswer` などのドメインモデルを定義します。
- `parser.py`
  - `pdfplumber`と`PyMuPDF`を使ってPDFを解析し、テキスト、表、図版のraw chunkを作ります。
- `pipeline.py`
  - 標準のsequential pipelineを実装します。
- `langgraph_pipeline.py`
  - LangGraphのstate graphベースでingest / queryを実装します。
- `crewai_pipeline.py`
  - CrewAI経路のingest / queryラッパーを実装します。
- `store.py`
  - ChromaDBへのupsert / retrieveを扱います。
- `cache.py`
  - モデルのメモリ内キャッシュと、`HF_HOME` の実行時設定を扱います。

### 2.2 `src/agents`

- `extraction.py`
  - `TextAgent`, `TableAgent`, `VisionAgent`
- `router.py`
  - `RawChunk.chunk_type` を見て抽出エージェントへ振り分けます。
- `validation.py`
  - `ChunkValidatorAgent`, `AnswerValidatorAgent`
- `orchestrator.py`
  - `ReasoningOrchestratorAgent`
- `crewai_agents.py`
  - CrewAIから使うagent定義群

### 2.3 `src/integrations`

- `langfuse.py`
  - trace, span, scoreを送信します。設定がなければno-opとして動きます。
- `dspy_adapter.py`, `dspy_modules.py`
  - DSPyとanswer validationの接続を担当します。
- `crew_mlx_tools.py`
  - CrewAIからローカルMLXエージェントを呼ぶためのtoolkitです。

### 2.4 `src/utils`

- `serialization.py`
  - chunks / answer JSONを`./output`に保存します。
- `audit.py`
  - 監査用JSON / HTML / 画像を出力します。

## 3. 実行モード

### 3.1 Sequential

標準経路です。`src/core/pipeline.py`の`AgenticRAGPipeline`がingestとqueryを担当します。

ingestでは次の順に進みます。

1. `PDFParser.parse()`
2. `AgentRouter.route()` による抽出
3. 任意のchunk validation
4. `ChunkStore.upsert()`
5. 任意の監査出力

queryでは次の順に進みます。

1. retrieve
2. orchestrator generate
3. 任意のanswer validation

### 3.2 LangGraph

`src/core/langgraph_pipeline.py`に2系統あります。

- `LangGraphIngestPipeline`
- `LangGraphQueryPipeline`

特徴はノード遷移が明示されていることです。とくにquery側はretrieve、quality check、generate、validate、revise、finalizeのフェーズがstate graphに分かれています。

CLI上の適用範囲は次の通りです。

- `ingest --use-langgraph`: ingestに適用
- `query --use-langgraph`: queryに適用
- `pipeline --use-langgraph`: queryフェーズのみに適用

### 3.3 CrewAI

CrewAI経路は`src/core/pipeline.py`と`src/core/crewai_pipeline.py`の組み合わせで動きます。

ただし、現行実装は「全面的にCrewAIが抽出を実行する」構成ではありません。

- ingest側
  - `AgenticRAGPipeline.ingest_with_crewai()` から `CrewAIIngestionPipeline.process_chunks()` へ入ります。
  - その中の`ExtractionCrew.extract_chunks()`は、Crew taskを使った抽出を意図的にスキップし、`AgentRouter`を直接呼んでローカルMLX抽出を行います。
  - `ValidationCrew`は現状、QA agentがなければそのまま通し、QA agentがあっても簡略実装です。
  - `LinkingCrew`はプレースホルダーに近い簡易linkingです。
- query側
  - `RAGQueryCrew`がretrieval、reasoning、verificationの流れをまとめます。
  - 失敗時はstandard queryにフォールバックします。

このため、CrewAIは「ローカルMLXエージェントを包むorchestration経路」であり、抽出・検証ロジックの本体を完全に置き換えてはいません。

## 4. データフロー

### 4.1 Ingest

1. `PDFParser.parse()` が `RawChunk` を返す
2. 抽出フェーズが `ProcessedChunk` を生成する
3. `validates=True`ならCHECKPOINT Aを実行する
4. 採用チャンクだけを `ChunkStore.upsert()` する
5. 監査出力が指定されていれば `save_chunk_audit()` を呼ぶ

SequentialとLangGraphはchunk validatorを実際に使いますが、CrewAI ingestのvalidationは現時点では簡略です。

### 4.2 Query

1. retrieve
2. orchestrator generate
3. `validates=True`ならCHECKPOINT B
4. 必要に応じてrevised answerへ置換
5. `RAGAnswer` を返す

LangGraph queryではvalidateの結果に応じてreviseノードを通ることがあります。

## 5. Parser の方針

### 5.1 テーブル検出

`PDFParser` は precision-first です。

- まず `pdfplumber.find_tables()` の標準候補を使います。
- fallback の text strategy は以下で有効になります。
  - 標準候補が 0 件
  - `enable_figure_aware_fallback=True` かつ figure があるページ
- 誤検出抑制のために、figure overlap、caption cue、numeric ratio、long-cell ratio を見ます。

このため本文誤検出は抑えやすい一方で、borderless table の一部は取りこぼします。

### 5.2 OCR

`parser.py` は `src/core/config.py` 経由で OCR 設定を読みます。

- `ocr.engine`
- `ocr.default_lang`
- `ocr.japanese_lang`
- `ocr.fallback_lang`
- `ocr.config`

現在の `settings.example.json` の既定値は `easyocr` です。OCR の詳細な運用注意は [docs/CONFIG_SETUP.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/CONFIG_SETUP.md) にまとめています。

## 6. バリデーション

### 6.1 CHECKPOINT A

- 対象: 抽出済み chunk
- 実装: `ChunkValidatorAgent`
- 目的: 保存前の品質担保
- 挙動:
  - valid なら採用
  - corrected が返れば差し替えて採用
  - invalid かつ corrected なしは破棄

### 6.2 CHECKPOINT B

- 対象: 生成済み answer
- 実装: `AnswerValidatorAgent`
- 目的: grounding の確認と hallucination 検出
- 挙動:
  - grounded ならそのまま返す
  - revised answer があれば差し替える
  - revised answer がなければ警告 prefix を付ける

DSPy 連携は AnswerValidator 側で使われます。

## 7. モデル管理

### 7.1 ロード / アンロード

`BaseLoadableModel` 系の重いモデルは `with model:` パターンで都度 load / unload されます。

- orchestrator
- chunk validator
- answer validator

一方、抽出エージェントは build 時に初期化されます。

### 7.2 キャッシュ

`src/core/cache.py` は実行時に `HF_HOME` を `~/.models` に固定します。これは `settings.json` の `cache.cache_dir` より優先される実装です。

メモリ内キャッシュでは、同一モデルの再ロードを避けます。`cleanup_unused_models()` は未使用モデルディレクトリのクリーンアップを試みます。

## 8. 出力と監査

### 8.1 JSON 出力

`src/utils/serialization.py` は出力先を `./output` に固定しています。

- `save_chunks()`
  - `<pdf_stem>_chunks.json`
- `save_answer()`
  - `<pdf_stem>_answer.json`

### 8.2 監査出力

`save_chunk_audit()` は次を出力します。

- `<pdf_stem>_audit.json`
- `<pdf_stem>_audit.html`
- `pages/*.png`
- `figures/*.png`

この監査出力先は CLI の `--output` から渡されます。

## 9. 観測性

`LangfuseTracer` が trace / span / score を記録します。環境変数がなければ no-op で処理は継続します。

- ingest trace
- query trace
- chunk quality score
- answer grounding score

`session_id` は関連 query を Langfuse 上で束ねるために使います。

## 10. 実装上の注意

- `app.py` の CLI 既定モデル値は `settings.json` の model 設定より先に適用されます。
- `lazy_agents` は CLI から渡せますが、現時点では pipeline インスタンスに保持されるだけで、目立った実行分岐には接続されていません。
- `pipeline --use-langgraph` は ingest を LangGraph 化しません。
