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
- 埋め込みモデル: `kun432/cl-nagoya-ruri-large` (デフォルト, Ollama) または `intfloat/multilingual-e5-small` 等
- ベクトルストア: ChromaDB
- 主な処理:
  - PDF parse (テキスト / テーブル / 図版)
  - チャンク抽出と品質検証 (CHECKPOINT A)
  - ハイブリッド検索 (ベクトル検索 + BM25 キーワード検索、RRF でフュージョン)
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
| テキスト抽出 | `qwen3:8b` | 5.2 GB |
| テーブル抽出 | `qwen2.5:3b` | 1.9 GB |
| 図版抽出 (VLM) | `ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227` | 9B class (BF16) |
| チャンク検証 (VLM) | `qwen2.5vl:7b` | 6.0 GB |
| 推論オーケストレーター / 回答検証 / DSPy 検証 | `gemma4:latest` | depends on local Ollama manifest |

```bash
ollama pull qwen3:8b
ollama pull qwen2.5:3b
ollama pull ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227
ollama pull qwen2.5vl:7b
ollama pull gemma4:latest
```

注: Ricoh VLM は利用規約への同意が必要な場合があります。`ollama pull` が利用環境で失敗する場合は、`models.vision_extraction` を従来の `qwen2.5vl:7b` に戻して運用してください。

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
    "vision_extraction": "ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227",
    "chunk_validator": "qwen2.5vl:7b",
    "orchestrator": "gemma4:latest",
    "answer_validator": "gemma4:latest",
    "dspy_lm": "gemma4:latest",
    "embedder": "kun432/cl-nagoya-ruri-large"
  },
  "embedder": {
    "backend": "ollama",
    "query_prefix": null,
    "passage_prefix": null,
    "batch_size": 32
  }
}
```

デフォルトは `kun432/cl-nagoya-ruri-large`（Ollama バックエンド）です。
sentence-transformers を使う場合は `settings.example.json` を参考に切り替えてください。

詳細は [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md) を参照してください。

### 5. 任意の環境変数

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # または後方互換エイリアス LANGFUSE_HOST
```

Langfuse のキーが未設定でも処理は継続します。
Langfuse トレースにはモデル名として Ollama モデル名 (`qwen3:8b` 等) が記録されます。

### 6. OCR 前提

OCR のメインエンジンは **EasyOCR** です。`uv sync` または `pip install -e .` で自動インストールされます。

図版領域の補助 OCR に Tesseract が使われます。日本語 PDF を扱う場合は Tesseract の日本語データも必要です。

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs    # jpn が表示されれば OK
```

OCR エンジンは `settings.json` の `ocr.engine` で切り替えられます (既定: `easyocr`)。

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
python app.py ingest ./in/sample.pdf
python app.py query "図2は何を示していますか？"
python app.py pipeline ./in/sample.pdf "要点を3つで要約して"
```

### 実行モード

現在は Sequential のみサポートしています。

### 主要オプション

| オプション | 説明 |
|---|---|
| `--validate` / `--no-validate` | CHECKPOINT A (ingest) / CHECKPOINT B (query) の有効化 |
| `--enable-figure-aware-fallback` | parser の figure-aware テーブル fallback を有効化 |
| `--lazy-agents` | 抽出エージェントをチャンクごとにロード/アンロード (VRAM 節約、低速化) |
| `--session-id` | Langfuse トレースのグループ ID |
| `--storage-dir` | ChromaDB 保存先 |
| `--output` | 監査レポートの出力先 |
| `--text-model` など | ingest/query でモデルを上書き |

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `./out/<stem>_chunks.json` | 抽出チャンク一覧 |
| `./out/<stem>_answer.json` | 質問応答結果 |
| `./out/<stem>_audit.json` | 監査ログ |
| `./out/<stem>_audit.html` | 監査レポート (HTML) |
| `./out/<stem>_audit/pages/*.png` | ページプレビュー画像 |
| `./out/<stem>_audit/figures/*.png` | 図版画像 |

---

## アーキテクチャ概要

注: 下記のモデル ID は `settings.json` または CLI オプションで上書き可能です。既定値はセットアップ節と `src/core/config.py` の定義を参照してください。

```
app.py (CLI)
  └── pipeline.py
        ├── parser.py          — PDF parse (pdfplumber + PyMuPDF + OCR)
        ├── agents/
        │   ├── TextAgent      — qwen3:8b   テキスト抽出
        │   ├── TableAgent     — qwen2.5:3b テーブル抽出
        │   ├── VisionAgent    — ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227 図版解析 (VLM)
        │   ├── ChunkValidatorAgent   — qwen2.5vl:7b CHECKPOINT A
        │   ├── ReasoningOrchestratorAgent — gemma4:latest 回答生成
        │   └── AnswerValidatorAgent  — gemma4:latest (DSPy) CHECKPOINT B
        ├── store.py           — ChromaDB + BM25 ハイブリッド検索 (RRF)
        └── integrations/
            ├── langfuse.py    — Langfuse v3 OTel トレース
            └── dspy_adapter.py — DSPy + Ollama 設定ヘルパー
```

すべての LLM 呼び出しは `ollama.Client.chat()` 経由で行われます。
モデルのロード / アンロードおよび VRAM 管理は Ollama サーバーが担います。
埋め込みモデルは `ollama` バックエンド（デフォルト: `kun432/cl-nagoya-ruri-large`）
または `sentence_transformers` バックエンド（例: `intfloat/multilingual-e5-small`）から選択できます。

VLM の段階導入方針:
- Phase 1: `vision_extraction` を Ricoh VLM に置換
- Phase 2: 品質評価後に `chunk_validator` 置換を検討

---

## 設定に関する注意

- `src/core/config.py` の `_DEFAULTS` が全設定の基底値です。
- `settings.json` で値を上書きできます (deep merge)。
- `app.py` の CLI 引数は `settings.json` より優先されます。
  ただし `app.py` のモデル既定値は `config.get_model(...)` から取得されるため、
  CLI で `--text-model` 等を明示しない場合は `settings.json` の `models.*` が既定値として使われます。
  (`settings.json` が無い場合のみ `_DEFAULTS` にフォールバック)
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
- [docs/library.md](docs/library.md)
- [tests/README.md](tests/README.md)
