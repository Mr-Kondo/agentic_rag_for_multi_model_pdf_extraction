# Agentic RAG for Multi-Model PDF Extraction

Apple Silicon（MLX）向けに最適化した、PDF解析とRAG質問応答のローカル実行パイプラインです。

このプロジェクトは、PDFから抽出したテキスト・テーブル・図表チャンクをベクトル化し、質問に対して根拠付き回答を生成します。抽出時と回答時に2段階の検証を実行でき、LangGraph/CrewAIのモード切り替えにも対応しています。

## 現在の実装ステータス

- 実装言語: Python
- Python要件: 3.13以上
- 実行環境想定: macOS（Apple Silicon）
- 主要モード:
  - Sequential pipeline（標準）
  - LangGraph pipeline（ingest/queryで利用可能）
  - CrewAI integration（`--use-crewai`）

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

`settings.example.json`を複製して`settings.json`を作成し、必要に応じてモデルIDを調整します。

```bash
cp settings.example.json settings.json
```

### 3. 環境変数（任意）

`.env`の例:

```bash
HF_TOKEN=your_huggingface_token
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
HF_HOME=./models
```

Langfuseキーが未設定でも処理自体は継続します。

### 4. 日本語OCR前提（推奨）

日本語PDFの文字化け抑止には、Tesseractの日本語言語データが必要です。

```bash
# tesseract本体
brew install tesseract

# 追加言語データ（環境に応じてパッケージ名が異なる場合あり）
brew install tesseract-lang

# 利用可能言語の確認
tesseract --list-langs
```

`tesseract --list-langs`に`jpn`が含まれていない場合、`settings.json`で`ocr.default_lang`を`jpn+eng`にしていても日本語OCRは劣化します。

## CLI

エントリポイントは`app.py`です。`agentic-rag`スクリプトでも実行できます。

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

# ingest + queryを連続実行
python app.py pipeline ./input/sample.pdf "要点を3つで要約して"
```

### モード切り替え

```bash
# LangGraph ingest
python app.py ingest ./input/sample.pdf --use-langgraph

# LangGraph query
python app.py query "結論は？" --use-langgraph

# CrewAI（ingest/query/pipelineで使用可能）
python app.py ingest ./input/sample.pdf --use-crewai
python app.py query "図表間の関係は？" --use-crewai
```

### 主要オプション

- `--validate` / `--no-validate`
  - ingest: CHECKPOINT A（chunk validation）の有効/無効
  - query: CHECKPOINT B（answer validation）の有効/無効
- `--enable-figure-aware-fallback`
  - ingest/pipelineで有効
  - parserのtable fallback適用条件を拡張（下記ポリシー参照）
- `--use-langgraph`
  - `ingest`: LangGraphIngestPipelineを使用
  - `query`: LangGraphQueryPipelineを使用
  - `pipeline`: queryフェーズにのみ適用（ingestはSequential/CrewAI）
- `--use-crewai`
- `--session-id <id>`
- `--storage-dir <dir>`
- `--output <dir>`
- `--lazy-agents`
- `--text-model`, `--table-model`, `--vision-model`
- `--orchestrator-model`, `--chunk-validator-model`, `--answer-validator-model`

## モードの実装実態

- Sequential:
  - `src/core/pipeline.py`の標準経路
  - ingest/queryの両方を担当
- LangGraph:
  - `src/core/langgraph_pipeline.py`
  - ingest: `LangGraphIngestPipeline`を`ingest --use-langgraph`で使用
  - query: `LangGraphQueryPipeline`を`query --use-langgraph`で使用
  - pipelineコマンドではqueryフェーズのみ`--use-langgraph`が適用される
- CrewAI:
  - `src/core/crewai_pipeline.py` + `src/integrations/crew_mlx_tools.py`
  - `--use-crewai` ingest経路では、抽出フェーズのCrewタスク実行をスキップし、`AgentRouter`経由のローカルMLX抽出を使用
  - `--no-validate`指定時はCrewAI ingestのvalidationフェーズをスキップ

## 出力ファイル

標準出力ディレクトリは`./output`です。

代表的な出力:

- `<pdf_stem>_chunks.json`
- `<pdf_stem>_answer.json`
- `query_answer.json`（`query`コマンド実行時）
- `<pdf_stem>_audit.json`（監査出力有効時）
- `<pdf_stem>_audit.html`（監査出力有効時）
- `<pdf_stem>_audit/pages/*.png`（監査出力有効時）
- `<pdf_stem>_audit/figures/*.png`（監査出力有効時）

注意点:

- `save_chunks()`/`save_answer()`は`src/utils/serialization.py`の`OUTPUT_DIR`（`./output`）に保存します。
- `--output`は主に監査レポート（`save_chunk_audit`）の出力先として使われます。
- 現状のCLIログには`--output`へ保存されたように見える文言がありますが、chunks/answer JSONの実保存先は`./output`です。

## テーブル検出ポリシー（現状）

- parserはprecision-first方針です（デフォルト）。
- `pdfplumber.find_tables()`の標準候補を優先します。
- fallback（text strategy）の実行条件:
  - 標準候補が0件のページでは常に実行
  - `--enable-figure-aware-fallback`が有効で、かつページ内にfigureがある場合も実行
- 本文誤検出（proseをtableと判定）を抑えるため、数値セル率・長文セル率・キャプション手掛かりで追加フィルタリングします。
- そのため、borderless tableの一部は未検出になる可能性があります。図中表はVision/Table extraction経路で補完される設計です。

## モデル既定値（CLIデフォルト）

`app.py`の`DEFAULT_MODELS`:

- text: `mlx-community/Phi-3.5-mini-Instruct-4bit`
- text: `mlx-community/Qwen3-8B-4bit`
- table: `mlx-community/Qwen2.5-3B-Instruct-4bit`
- vision: `mlx-community/SmolVLM-256M-Instruct-4bit`
- orchestrator: `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit`
- chunk_validator: `mlx-community/Qwen2-VL-7B-Instruct-4bit`
- answer_validator: `mlx-community/Qwen3-8B-4bit`

`settings.example.json`は`app.py`の`DEFAULT_MODELS`と整合する値に揃えています。必要があればCLIオプションで一時的に上書きできます。

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
- OCR言語データ（特に`jpn`）が未導入の環境では、日本語抽出品質が低下します。
- テーブル検出はPDF構造の品質に依存します（precision-first設定のため、本文誤検出抑制を優先）。
- `pipeline --use-langgraph`は現状queryフェーズに適用されます（ingestフェーズはSequentialまたはCrewAI）。

## 関連ドキュメント

- `docs/ARCHITECTURE.md`: 現行アーキテクチャ詳細
- `docs/FLOW.md`: CLI処理フロー（Mermaid）
- `docs/CONFIG_SETUP.md`: 設定周り
- `tests/README.md`: テストガイド
