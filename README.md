# Agentic RAG for Multi-Model PDF Extraction

Ollama を使ってローカル実行する、PDF 抽出 + RAG 回答パイプラインです。

現在のサポート runtime は Sequential のみです。
ingest で PDF から text/table/figure を抽出して ChromaDB へ保存し、
query で根拠チャンクにもとづく回答を返します。

## 概要

- 対象環境: macOS (Apple Silicon / Intel)
- Python: 3.13+
- 推論バックエンド: Ollama
- ベクトルストア: ChromaDB
- 検索: dense + BM25 のハイブリッド検索 (RRF)
- 検証:
  - CHECKPOINT A: ingest 時の chunk 品質検証
  - CHECKPOINT B: query 時の grounding / hallucination 検証

## クイックスタート

1) 依存をインストール

```bash
uv sync
```

2) 設定ファイルを作成

```bash
cp settings.example.json settings.json
```

3) Ollama を起動し、必要モデルを pull

```bash
brew install ollama
ollama serve
```

```bash
ollama pull qwen3.5:latest
ollama pull qwen2.5:7b
ollama pull qwen3-vl:latest
ollama pull gemma4:latest
ollama pull kun432/cl-nagoya-ruri-large:latest
```

4) 実行

```bash
uv run python app.py ingest ./in/sample.pdf
uv run python app.py query "図2は何を示していますか？"
uv run python app.py pipeline ./in/sample.pdf "要点を3つで要約して"
```

## OCR エンジン

`settings.json` の `ocr.engine` で選択します。

- `easyocr`: 既定の OCR エンジン
- `yomitoku`: 日本語向け OCR。`ocr.yomitoku_device` と `ocr.prewarm_yomitoku` を利用
- `tesseract`: 代替 OCR エンジン

parser は region policy に応じて engine を選択し、`easyocr` / `yomitoku` が空結果の場合は
`tesseract` へフォールバックします。

Tesseract を使う場合は日本語データ導入が必要です。

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs
```

詳細は [docs/OCR_ENGINE_BEHAVIOR.md](docs/OCR_ENGINE_BEHAVIOR.md) を参照してください。

## CLI

```bash
uv run python app.py --help
```

### サブコマンド

- `ingest <pdf_path>`
- `query "<question>"`
- `pipeline <pdf_path> "<question>"`

### 主要オプション

| オプション | 説明 |
|---|---|
| `--validate` / `--no-validate` | ingest では CHECKPOINT A、query では CHECKPOINT B を制御 |
| `--enable-figure-aware-fallback` | ingest / pipeline の parser で figure-aware table fallback を有効化 |
| `--quality-mode` | 実行時に `pipeline.text_passthrough=False` を適用 |
| `--fast-mode` | 実行時に `pipeline.text_passthrough=True` を適用 |
| `--lazy-agents` | フラグは受理され pipeline に渡されるが、現時点では大きな挙動分岐は限定的 |
| `--session-id` | query / pipeline の Langfuse grouping ID |
| `--storage-dir` | ChromaDB 保存先 |
| `--output` | chunks / answer / audit の出力先 (既定: `./out`) |

正確な挙動は [docs/CLI_BEHAVIOR_REFERENCE.md](docs/CLI_BEHAVIOR_REFERENCE.md) を参照してください。

## 出力ファイル

`--output` は chunks / answer / audit すべてに適用されます。

| コマンド | 主な生成物 |
|---|---|
| ingest | `<pdf_stem>_chunks.json`, `<pdf_stem>_audit.json`, `<pdf_stem>_audit.html`, `pages/*.png`, `figures/*.png` |
| query | `query_answer.json` |
| pipeline | `<pdf_stem>_chunks.json`, `<pdf_stem>_answer.json`, `<pdf_stem>_audit.json`, `<pdf_stem>_audit.html` |

注: query は実装上 `query.pdf` を保存名の基準に使うため、answer は `query_answer.json` になります。

## 設定優先順位

1. CLI 引数
2. `settings.json`
3. `src/core/config.py` の `_DEFAULTS`

補足:

- モデル系 CLI 既定値は `config.get_model(...)` で解決されるため、`settings.json` があれば
  help の default 表示にも反映されます。
- `settings.example.json` は推奨構成例であり、`_DEFAULTS` と異なる値を含みます。

## 関連ドキュメント

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md)
- [docs/FLOW.md](docs/FLOW.md)
- [docs/library.md](docs/library.md)
- [docs/CLI_BEHAVIOR_REFERENCE.md](docs/CLI_BEHAVIOR_REFERENCE.md)
- [docs/OCR_ENGINE_BEHAVIOR.md](docs/OCR_ENGINE_BEHAVIOR.md)
- [docs/KNOWN_CAVEATS.md](docs/KNOWN_CAVEATS.md)
