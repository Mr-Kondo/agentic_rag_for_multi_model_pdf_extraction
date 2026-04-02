# Agentic RAG for Multi-Model PDF Extraction

Apple Silicon上で動かすことを前提にした、ローカルPDF解析とRAG質問応答のパイプラインです。

PDFから抽出したテキスト、テーブル、図版をチャンク化してChromaDBに保存し、質問に対して根拠付き回答を返します。抽出時と回答時には任意で2段階のバリデーションを実行でき、Sequential、LangGraph、CrewAIの経路を切り替えられます。

## 概要

- 対象環境: macOS / Apple Silicon
- Python: 3.13以上
- 主な処理:
  - PDF parse
  - テキスト、表、図版の抽出
  - チャンク品質検証
  - ベクトル検索
  - 根拠性付き回答生成
- 実行モード:
  - Sequential
  - LangGraph
  - CrewAI

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

または:

```bash
pip install -e .
```

### 2. 設定ファイルの用意

```bash
cp settings.example.json settings.json
```

`settings.json`はOCR設定、validation閾値、pipeline設定などの既定値に使われます。詳細は [docs/CONFIG_SETUP.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/CONFIG_SETUP.md) を参照してください。

### 3. 任意の環境変数

```bash
HF_TOKEN=your_huggingface_token
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Langfuseのキーが未設定でも処理は継続します。

### 4. OCR 前提

現在の`settings.example.json`ではOCRエンジン既定値は`easyocr`です。日本語PDFを扱う場合はEasyOCRを前提にしつつ、Tesseractを補助的に入れておくと切り分けしやすくなります。

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs
```

`jpn`が表示されない環境では、Tesseract側にフォールバックしたときの日本語品質が落ちます。

## CLI

エントリポイントは `app.py` です。インストール後は `agentic-rag` でも実行できます。

```bash
python app.py --help
agentic-rag --help
```

### サブコマンド

- `ingest <pdf_path>`
- `query "<question>"`
- `pipeline <pdf_path> "<question>"`

### 基本例

```bash
python app.py ingest ./input/sample.pdf
python app.py query "図2は何を示していますか？"
python app.py pipeline ./input/sample.pdf "要点を3つで要約して"
```

### 実行モード

```bash
python app.py ingest ./input/sample.pdf --use-langgraph
python app.py query "結論は？" --use-langgraph
python app.py ingest ./input/sample.pdf --use-crewai
python app.py query "図表間の関係は？" --use-crewai
```

現行実装での適用範囲は次の通りです。

- Sequential:
  - 標準経路です。
  - ingestとqueryの両方を`src/core/pipeline.py`で処理します。
- LangGraph:
  - `ingest --use-langgraph` では `LangGraphIngestPipeline` を使います。
  - `query --use-langgraph` では `LangGraphQueryPipeline` を使います。
  - `pipeline --use-langgraph`ではqueryフェーズにのみ適用されます。
- CrewAI:
  - `--use-crewai`はingest、query、pipelineで使えます。
  - ingest側ではCrewAI経路に入っても、抽出そのものはExtraction crewを全面実行せず、ローカル`AgentRouter`によるMLX抽出を使います。
  - query側ではCrewAIのretrieval / reasoning / verificationラッパーを使います。
- `pipeline` で `--use-crewai` と `--use-langgraph` を同時に指定した場合:
  - ingestはCrewAI優先
  - queryもCrewAI優先

### 主要オプション

- `--validate` / `--no-validate`
  - ingest: CHECKPOINT A
  - query: CHECKPOINT B
  - pipeline: 両方
- `--use-langgraph`
  - ingestとqueryで有効
  - pipelineではqueryフェーズのみ有効
- `--use-crewai`
  - ingest、query、pipelineで有効
- `--enable-figure-aware-fallback`
  - ingest / pipelineのparserにだけ効きます
- `--session-id`
  - Langfuseのtrace groupingに使います
- `--storage-dir`
  - ChromaDB保存先
- `--output`
  - 監査レポートの出力先として使われます
- `--lazy-agents`
  - CLIにはありますが、現状はpipelineインスタンスに保持されるだけで、明確な追加分岐にはまだ使われていません
- `--text-model`, `--table-model`, `--vision-model`
- `--orchestrator-model`, `--chunk-validator-model`, `--answer-validator-model`

## 出力

代表的な出力物は次の通りです。

- `./output/<pdf_stem>_chunks.json`
- `./output/<pdf_stem>_answer.json`
- `./output/query_answer.json`相当のquery出力
- `<output_dir>/<pdf_stem>_audit.json`
- `<output_dir>/<pdf_stem>_audit.html`
- `<output_dir>/<pdf_stem>_audit/pages/*.png`
- `<output_dir>/<pdf_stem>_audit/figures/*.png`

現行実装では少し癖があります。

- JSON保存は`src/utils/serialization.py`の`OUTPUT_DIR`に固定されており、実体は`./output`です。
- `--output`は主に監査レポートの出力先です。
- CLIのログは`--output`配下に保存されたように見える文言を出しますが、chunks / answer JSONの保存先そのものは切り替わりません。
- `app.py`の既定値では`--output`に`./output`が入るため、通常実行でもJSON保存処理は走ります。

## 設定に関する注意

`settings.json`は万能の設定入口ではありません。現行のCLI実装では、モデル選択まわりに次の制約があります。

- `src/core/config.py`には`settings.json`のモデル既定値があります。
- ただし`app.py`のCLI引数にもモデル既定値がハードコードされており、CLI経由で実行すると、未指定時でも`app.py`側の既定値が優先されます。
- そのため、`settings.json`の`models.*`を変えても、現状の`python app.py ...`では自動反映されない項目があります。
- 一方でOCR設定、validation設定、parser設定などは`settings.json`側の値が参照されます。

## パーサの現状

- table検出はprecision-firstです。
- `pdfplumber.find_tables()` の標準候補を優先します。
- fallbackのtext strategyは次の条件で走ります。
  - 標準候補が0件
  - `--enable-figure-aware-fallback`が有効で、かつページにfigureがある
- 本文の誤検出を避けるため、数値セル率、長文セル率、caption cue、figure重なりで追加フィルターをかけます。

## 制約と既知事項

- MLX前提のため、macOS / Apple Siliconを主対象にしています。
- `src/core/cache.py` は実行時に `HF_HOME` を `~/.models` へ固定します。
- `settings.json` の `cache.cache_dir` は現状の実装では保存先を切り替えていません。
- OCRの既定値はEasyOCRですが、OCR品質はPDFの状態とレイアウトに左右されます。
- `pipeline --use-langgraph`はingestには効きません。

## テスト

```bash
pytest tests/ -v
pytest tests/test_parser.py -v
```

## 関連ドキュメント

- [docs/ARCHITECTURE.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/ARCHITECTURE.md)
- [docs/CONFIG_SETUP.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/CONFIG_SETUP.md)
- [docs/FLOW.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/FLOW.md)
- [tests/README.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/tests/README.md)
