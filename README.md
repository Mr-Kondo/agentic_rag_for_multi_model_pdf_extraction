# Agentic RAG for Multi-Model PDF Extraction

Ollama を使ってローカルで動かす、マルチモーダル PDF 解析と RAG 質問応答のパイプラインです。

PDF からテキスト・テーブル・図版を抽出してチャンク化し、ChromaDB に保存します。
質問に対しては根拠付き回答を返します。抽出時と回答時に 2 段階のバリデーションを実行でき、
標準の Sequential パイプラインで ingest/query を実行します。

---

## 概要

- 対象環境: macOS (Apple Silicon / Intel)
- Python: 3.13 以上
- 推論バックエンド: [Ollama](https://ollama.com) (すべての LLM をローカルサーバーで提供)
- 埋め込みモデル: `intfloat/multilingual-e5-small` (HuggingFace sentence-transformers)
- ベクトルストア: ChromaDB
- 主な処理:
  - PDF parse (テキスト / テーブル / 図版)
  - チャンク抽出と品質検証 (CHECKPOINT A)
  - ベクトル検索
  - 根拠付き回答生成と幻覚検出 (CHECKPOINT B)
  - Langfuse によるオブザーバビリティ
- 実行モード: Sequential

---

## セットアップ

### 1. Ollama のインストール

```bash
brew install ollama
ollama serve          # 別ターミナルで常時起動しておく
```

### 2. 必要モデルのダウンロード

| 役割 | モデル | サイズ概算 |
|---|---|---|
| テキスト抽出 / 回答検証 | `qwen3:8b` | 5.2 GB |
| テーブル抽出 | `qwen2.5:3b` | 1.9 GB |
| 図版抽出 / チャンク検証 (VLM) | `qwen2.5vl:7b` | 6.0 GB |
| 推論オーケストレーター | `deepseek-r1:8b` | 5.2 GB |
| DSPy 幻覚検出 | `qwen2.5:7b` | 4.7 GB |

```bash
ollama pull qwen3:8b
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:7b
ollama pull deepseek-r1:8b
ollama pull qwen2.5:7b
```

### 3. Python 依存関係のインストール

```bash
uv sync
```

または:

```bash
pip install -e .
```

### 4. 設定ファイルの用意

```bash
cp settings.example.json settings.json
```

`settings.json` で Ollama エンドポイント URL やモデル名を変更できます。

```json
{
  "ollama": {
    "base_url": "http://localhost:11434"
  },
  "models": {
    "text_extraction": "qwen3:8b",
    "table_extraction": "qwen2.5:3b",
    "vision_extraction": "qwen2.5vl:7b",
    "chunk_validator": "qwen2.5vl:7b",
    "orchestrator": "deepseek-r1:8b",
    "answer_validator": "qwen3:8b",
    "dspy_lm": "qwen2.5:7b",
    "embedder": "intfloat/multilingual-e5-small"
  }
}
```

詳細は [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md) を参照してください。

### 5. 任意の環境変数

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Langfuse のキーが未設定でも処理は継続します。
Langfuse トレースにはモデル名として Ollama モデル名 (`qwen3:8b` 等) が記録されます。

### 6. OCR 前提

日本語 PDF を扱う場合は Tesseract の日本語データをインストールしてください。

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs    # jpn が表示されれば OK
```

---

## CLI

エントリポイントは `app.py` です。インストール後は `agentic-rag` でも呼び出せます。

```bash
python app.py --help
agentic-rag --help
```

### サブコマンド

- `ingest <pdf_path>` — PDF を解析してベクトルストアへ保存
- `query "<question>"` — 保存済みチャンクから根拠付き回答を生成
- `pipeline <pdf_path> "<question>"` — ingest + query を一括実行

### 基本例

```bash
python app.py ingest ./input/sample.pdf
python app.py query "図2は何を示していますか？"
python app.py pipeline ./input/sample.pdf "要点を3つで要約して"
```

### 実行モード

現在は Sequential のみサポートしています。

### 主要オプション

| オプション | 説明 |
|---|---|
| `--validate` / `--no-validate` | CHECKPOINT A (ingest) / CHECKPOINT B (query) の有効化 |
| `--enable-figure-aware-fallback` | parser の figure-aware テーブル fallback を有効化 |
| `--session-id` | Langfuse トレースのグループ ID |
| `--storage-dir` | ChromaDB 保存先 |
| `--output` | 監査レポートの出力先 |
| `--text-model` など | ingest/query でモデルを上書き |

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `./output/<stem>_chunks.json` | 抽出チャンク一覧 |
| `./output/<stem>_answer.json` | 質問応答結果 |
| `./output/<stem>_audit.json` | 監査ログ |
| `./output/<stem>_audit.html` | 監査レポート (HTML) |
| `./output/<stem>_audit/pages/*.png` | ページプレビュー画像 |
| `./output/<stem>_audit/figures/*.png` | 図版画像 |

---

## アーキテクチャ概要

```
app.py (CLI)
  └── pipeline.py
        ├── parser.py          — PDF parse (pdfplumber + PyMuPDF + OCR)
        ├── agents/
        │   ├── TextAgent      — qwen3:8b   テキスト抽出
        │   ├── TableAgent     — qwen2.5:3b テーブル抽出
        │   ├── VisionAgent    — qwen2.5vl:7b 図版解析 (VLM)
        │   ├── ChunkValidatorAgent   — qwen2.5vl:7b CHECKPOINT A
        │   ├── ReasoningOrchestratorAgent — deepseek-r1:8b 回答生成
        │   └── AnswerValidatorAgent  — qwen2.5:7b (DSPy) CHECKPOINT B
        ├── store.py           — ChromaDB (intfloat/multilingual-e5-small)
        └── integrations/
            ├── langfuse.py    — Langfuse v3 OTel トレース
            └── dspy_adapter.py — DSPy + Ollama 設定ヘルパー
```

すべての LLM 呼び出しは `ollama.Client.chat()` 経由で行われます。
モデルのロード / アンロードおよび VRAM 管理は Ollama サーバーが担います。
埋め込みモデル (`intfloat/multilingual-e5-small`) のみ HuggingFace からダウンロードし、
`sentence-transformers` で推論します (キャッシュ先: `~/.models`)。

---

## 設定に関する注意

- `src/core/config.py` の `_DEFAULTS` が全設定の基底値です。
- `settings.json` で値を上書きできます (deep merge)。
- `app.py` の CLI 引数は `settings.json` より優先されます。
  そのため `settings.json` の `models.*` を変えても、CLI で `--text-model` 等を
  明示しない限り、`app.py` 側のデフォルト値が使われる場合があります。
- OCR / バリデーション / パーサ設定は `settings.json` が参照されます。

---

## パーサの現状

- テーブル検出は precision-first です。
- `pdfplumber.find_tables()` の標準候補を優先します。
- fallback の text strategy は標準候補 0 件時や figure-aware fallback 有効時に実行されます。
- 数値セル率・長文セル率・caption cue・figure 重なりで誤検出を抑制します。

---

## テスト

```bash
pytest tests/ -v
pytest tests/test_parser.py -v
```

---

## 関連ドキュメント

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md)
- [docs/FLOW.md](docs/FLOW.md)
- [tests/README.md](tests/README.md)
