# Agentic RAG for Multi-Model PDF Extraction

Apple Silicon (MLX) 向けに最適化した、PDF解析とRAG質問応答のローカル実行パイプラインです。

このプロジェクトは、PDFから抽出したテキスト・テーブル・図表チャンクをベクトル化し、質問に対して根拠付き回答を生成します。抽出時と回答時に2段階の検証を実行でき、LangGraph/CrewAIのモード切り替えにも対応しています。

## 現在の実装ステータス

- 実装言語: Python
- Python要件: 3.13以上
- 実行環境想定: macOS (Apple Silicon)
- 主要モード:
- Sequential pipeline（標準）
- LangGraph pipeline（ingest/queryで利用可能）
- CrewAI integration (`--use-crewai`)

## 主な機能

- マルチモーダル抽出:
- TextAgent、TableAgent、VisionAgentによるチャンク処理
- 2段階バリデーション:
- CHECKPOINT A: チャンク品質検証
- CHECKPOINT B: 回答の根拠性検証
- 3つの実行モード:
- Sequential
- LangGraph（ingest/query対応）
- CrewAI
- ChromaDBベースのセマンティック検索
- Langfuseトレーシング（環境変数設定時のみ有効）
- DSPy統合（AnswerValidator側）

## セットアップ

### 1. インストール

```bash
uv sync
```

または:

```bash
pip install -e .
```

### 2. 設定ファイル

`settings.example.json` を複製して `settings.json` を作成し、必要に応じてモデルIDを調整します。

```bash
cp settings.example.json settings.json
```

### 3. 環境変数 (任意)

`.env` の例:

```bash
HF_TOKEN=your_huggingface_token
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
HF_HOME=./models
```

Langfuseのキーが未設定でも処理自体は継続します。

## CLI

エントリポイントは `app.py` です。`agentic-rag` スクリプトでも実行できます。

```bash
python app.py --help
agentic-rag --help
```

### コマンド一覧

- `ingest <pdf_path>`
- `query "<question>"`
- `pipeline <pdf_path> "<question>"`

### 基本例

```bash
# PDF取り込み
python app.py ingest ./input/sample.pdf --validate

# 質問応答
python app.py query "図2は何を示していますか？" --validate

# ingest + query を連続実行
python app.py pipeline ./input/sample.pdf "要点を3つで要約して"
```

### モード切り替え

```bash
# LangGraph ingest
python app.py ingest ./input/sample.pdf --use-langgraph

# LangGraph query
python app.py query "結論は？" --use-langgraph

# CrewAI (ingest/query/pipelineで使用可能)
python app.py ingest ./input/sample.pdf --use-crewai
python app.py query "図表間の関係は？" --use-crewai
```

### 主要オプション

- `--validate` / `--no-validate`
- `--use-langgraph`（ingestとqueryでLangGraphを使用。pipelineではqueryフェーズに適用）
- `--use-crewai`
- `--session-id <id>`
- `--storage-dir <dir>`
- `--output <dir>`
- `--lazy-agents`
- `--text-model`, `--table-model`, `--vision-model`
- `--orchestrator-model`, `--chunk-validator-model`, `--answer-validator-model`

## モードの実装実態

- Sequential:
- `src/core/pipeline.py` の標準経路
- ingest/queryの両方を担当
- LangGraph:
- `src/core/langgraph_pipeline.py`
- ingest: `LangGraphIngestPipeline` を `ingest --use-langgraph` で使用
- query: `LangGraphQueryPipeline` を `query --use-langgraph` で使用
- pipelineコマンドではqueryフェーズのみ `--use-langgraph` が適用される
- CrewAI:
- `src/core/crewai_pipeline.py` + `src/integrations/crew_mlx_tools.py`
- `--use-crewai` 経路では、外部API依存を避けるために一部Crewをスキップし、ローカルMLX処理を優先する設計

## 出力ファイル

標準出力ディレクトリは `./output` です。

代表的な出力:

- `<pdf_stem>_chunks.json`
- `<pdf_stem>_answer.json`
- `query_answer.json`（`query` コマンド実行時）
- `<pdf_stem>_audit.json`（監査出力有効時）
- `<pdf_stem>_audit.html`（監査出力有効時）
- `<pdf_stem>_audit/pages/*.png`（監査出力有効時）
- `<pdf_stem>_audit/figures/*.png`（監査出力有効時）

注意点:

- `save_chunks()` / `save_answer()` は `src/utils/serialization.py` の `OUTPUT_DIR` (`./output`) に保存します。
- `--output` は主に監査レポート (`save_chunk_audit`) の出力先として使われます。
- CLIのログ文言上は `--output` へ保存されるように見える箇所がありますが、chunks/answer JSON の実保存先は `./output` です。

## モデル既定値 (CLIデフォルト)

`app.py` の `DEFAULT_MODELS`:

- text: `mlx-community/Phi-3.5-mini-Instruct-4bit`
- table: `mlx-community/Qwen2.5-3B-Instruct-4bit`
- vision: `mlx-community/SmolVLM-256M-Instruct-4bit`
- orchestrator: `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit`
- chunk_validator: `mlx-community/Qwen2-VL-7B-Instruct-4bit`
- answer_validator: `mlx-community/Qwen3-8B-4bit`

`settings.example.json` は `app.py` の `DEFAULT_MODELS` と整合する値に揃えています。必要があればCLIオプションで一時的に上書きできます。

## テスト

```bash
pytest tests/ -v
```

個別実行例:

```bash
pytest tests/test_langgraph_pipeline.py -v
pytest tests/test_dspy_validator.py -v
```

## 制約と既知事項

- MLX依存のため、Apple Silicon前提の設計です。
- OCRは自動起動しません。スキャンPDFでは前処理が必要です。
- テーブル検出はPDF構造の品質に依存します。
- `pipeline --use-langgraph` は現状 query フェーズに適用されます（ingest フェーズは Sequential または CrewAI）。

## 関連ドキュメント

- `docs/ARCHITECTURE.md`: 現行アーキテクチャ詳細
- `docs/CONFIG_SETUP.md`: 設定周り
- `tests/README.md`: テストガイド
