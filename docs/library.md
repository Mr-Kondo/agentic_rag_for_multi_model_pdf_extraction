# ライブラリ・フレームワーク解説

Last updated: 2026-04-19

このドキュメントは、Agentic RAG for Multi-Model PDF Extraction が依存する主要なフレームワーク・ライブラリについて説明します。
各項目では「概要」「このアプリでの用途」「役割」を記述します。

アーキテクチャ全体については [ARCHITECTURE.md](ARCHITECTURE.md) を、設定方法については [CONFIG_SETUP.md](CONFIG_SETUP.md) を参照してください。

---

## 目次

1. [LLM 推論 / エージェント](#1-llm-推論--エージェント)
   - [1.1 Ollama](#11-ollama)
   - [1.2 DSPy](#12-dspy)
2. [ベクトル検索 / 埋め込み](#2-ベクトル検索--埋め込み)
   - [2.1 ChromaDB](#21-chromadb)
   - [2.2 sentence-transformers](#22-sentence-transformers)
   - [2.3 rank-bm25](#23-rank-bm25)
3. [PDF 解析 / OCR](#3-pdf-解析--ocr)
   - [3.1 pdfplumber](#31-pdfplumber)
   - [3.2 PyMuPDF (fitz)](#32-pymupdf-fitz)
   - [3.3 EasyOCR](#33-easyocr)
   - [3.4 yomitoku](#34-yomitoku)
   - [3.5 pytesseract](#35-pytesseract)
   - [3.6 OpenCV](#36-opencv)
   - [3.7 Pillow](#37-pillow)
4. [オブザーバビリティ](#4-オブザーバビリティ)
   - [4.1 Langfuse](#41-langfuse)
5. [データモデル / ユーティリティ](#5-データモデル--ユーティリティ)
   - [5.1 Pydantic](#51-pydantic)
   - [5.2 tiktoken](#52-tiktoken)
   - [5.3 python-dotenv](#53-python-dotenv)

---

## 1. LLM 推論 / エージェント

### 1.1 Ollama

- **パッケージ**: `ollama>=0.4.0`
- **公式サイト**: https://ollama.com

**概要**

Ollama はローカルマシン上で大規模言語モデル (LLM) や視覚言語モデル (VLM) を動かすためのランタイムです。
Python SDK を通じて推論 API にアクセスできます。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/agents/extraction.py` | `TextAgent` / `TableAgent` / `VisionAgent` がそれぞれのチャンク種別に対応したモデルを呼び出し、構造化された JSON レスポンスを取得する |
| `src/agents/validation.py` | `ChunkValidatorAgent` が CHECKPOINT A において抽出品質を検証する際に VLM を呼び出す |
| `src/agents/orchestrator.py` | `ReasoningOrchestratorAgent` が取得チャンクをコンテキストとして与え、最終回答を生成する |
| `src/core/embedder.py` | `OllamaEmbedder` が Ollama の Embed API を利用してテキストをベクトル化する (オプション backend) |

**役割**

外部クラウド API に依存せず、すべての LLM 推論をローカルで完結させるための基盤です。
モデルのロード / アンロードも Ollama サーバーが管理するため、アプリ側は `with model:` ブロックで使用フェーズを限定するだけでメモリ効率を確保できます。

---

### 1.2 DSPy

- **パッケージ**: `dspy-ai>=2.5.0`
- **公式サイト**: https://dspy.ai

**概要**

DSPy は LLM プログラムを宣言的に記述するフレームワークです。
プロンプトを文字列で管理する代わりに、入出力の型付きシグネチャ (`dspy.Signature`) と推論モジュール (`dspy.ChainOfThought` など) を使って LLM の振る舞いを定義します。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/integrations/dspy_modules.py` | `AnswerGroundingSignature` として回答グラウンディング検証の入出力スキーマを定義する |
| `src/integrations/dspy_adapter.py` | Ollama バックエンドを DSPy の LM として設定する (`configure_ollama_lm`) |
| `src/agents/validation.py` | `AnswerValidatorAgent` の CHECKPOINT B で `dspy.ChainOfThought` を使って体系的なハルシネーション検出を行う |

**役割**

回答バリデーションに構造化出力と連鎖思考 (Chain-of-Thought) を組み合わせることで、ハルシネーション検出の精度と一貫性を高めます。
Pydantic と組み合わせて型安全な出力スキーマ (`AnswerGroundingOutput`) を保証します。

---

## 2. ベクトル検索 / 埋め込み

### 2.1 ChromaDB

- **パッケージ**: `chromadb>=1.5.1`
- **公式サイト**: https://www.trychroma.com

**概要**

ChromaDB はオープンソースのベクトルデータベースです。
ローカルファイルシステムへの永続保存をサポートしており、外部サービスなしで完結します。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/store.py` | `ChunkStore` が `PersistentClient` を使って `./chroma_db` 配下に `agentic_rag` コレクションを作成・管理する |

主な操作:
- **upsert**: `ProcessedChunk` のテキスト + サマリーをベクトル化して格納する
- **query**: クエリベクトルに対してコサイン類似度で近傍チャンクを検索する
- **peek / get**: BM25 インデックス再構築や埋め込み次元の整合性チェックに使用する

**役割**

PDF から抽出・変換されたチャンクを永続的に保管し、クエリ時にセマンティック検索を可能にするベクトルストアです。
内部的には HNSW (コサイン空間) を使用しています。

---

### 2.2 sentence-transformers

- **パッケージ**: `sentence-transformers>=5.2.3`
- **公式サイト**: https://www.sbert.net

**概要**

Hugging Face が提供するテキスト埋め込みライブラリです。
BERT 系モデルを用いて文・段落を高品質なベクトルに変換します。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/embedder.py` | `SentenceTransformerEmbedder` として実装され、`embedder.backend="sentence_transformers"` のときに使用される |

現行 default backend は `ollama` です。`sentence_transformers` はオプション backend として利用します。
切り替え時は再 ingest を推奨します。

**役割**

ローカルで動作する高品質な埋め込みモデルを提供します。
主用途は、Ollama backend が使えない環境や Hugging Face 由来モデルを直接利用したいケースです。

---

### 2.3 rank-bm25

- **パッケージ**: `rank-bm25>=0.2.2`
- **公式サイト**: https://github.com/dorianbrown/rank_bm25

**概要**

BM25 (Best Match 25) はキーワードベースの情報検索アルゴリズムです。
`rank-bm25` はその Python 実装で、`BM25Okapi` クラスを提供します。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/store.py` | `ChunkStore` がベクトル検索と並行して BM25 によるキーワード検索を実行し、Reciprocal Rank Fusion (RRF) で結果を統合する |

upsert のたびに ChromaDB の全ドキュメントから BM25 インデックスを再構築します。

**役割**

セマンティック検索だけでは拾いにくい固有名詞・専門用語を補完するスパース検索を提供します。
ベクトル検索 (dense) と BM25 (sparse) を RRF で融合したハイブリッド検索により、検索精度を向上させます。

---

## 3. PDF 解析 / OCR

### 3.1 pdfplumber

- **パッケージ**: `pdfplumber>=0.11.9`
- **公式サイト**: https://github.com/jsvine/pdfplumber

**概要**

pdfplumber は PDF からテキストブロックや表を高精度に抽出するライブラリです。
セル境界線の認識やネストした表の処理に優れています。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | `PDFParser` の主要エンジンとして、各ページのテキストブロック抽出と表の検出 (`find_tables`) に使用する |

`PDFParser` は precision-first の方針で pdfplumber の標準候補を優先し、候補がない場合や figure-aware fallback が有効な場合にテキスト戦略へ切り替えます。
誤検出抑制のために figure overlap / caption cue / numeric ratio / long-cell ratio のヒューリスティックを適用します。

**役割**

PDF の構造情報 (座標・セル境界) を活用した精度の高いテキスト・表抽出を担います。

---

### 3.2 PyMuPDF (fitz)

- **パッケージ**: `pymupdf>=1.27.1`
- **公式サイト**: https://pymupdf.readthedocs.io
- **インポート名**: `import pymupdf` (旧称 `fitz`)

**概要**

MuPDF エンジンをベースにした高速 PDF/XPS 処理ライブラリです。
ページのラスタライズや埋め込み画像の抽出が得意です。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | `PDFParser` が PDF ページを画像化 (ラスタライズ) して OCR にかけたり、PDF に埋め込まれた図・画像を `RawChunk(chunk_type=FIGURE)` として取り出したりする |

**役割**

pdfplumber が担うテキスト/表抽出を補完し、図・画像チャンクの抽出とページ全体のレンダリングを提供します。

---

### 3.3 EasyOCR

- **パッケージ**: `easyocr>=1.7.2`
- **公式サイト**: https://github.com/JaidedAI/EasyOCR

**概要**

ディープラーニングベースの OCR ライブラリです。
日本語を含む 80 以上の言語をサポートし、行ごとに信頼スコアを返します。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | `ocr.engine` / `region_policies.*.engine` で `easyocr` が選ばれたときに使用される |

信頼スコアが低い行は region policy に従って再 OCR 対象になります。
`ocr.prewarm_easyocr=true` で parser 初期化時にプリウォームします。

**役割**

テキスト抽出品質が低いページや、スキャン由来の画像 PDF に対して高精度な日本語 OCR を提供します。

---

### 3.4 yomitoku

- **パッケージ**: `yomitoku>=0.12.0`
- **公式サイト**: https://github.com/kotaro-kinoshita/yomitoku

**概要**

yomitoku は日本語向け OCR ライブラリです。`OCR(device=...)` で実行デバイスを指定できます。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | `ocr.engine` または `region_policies.*.engine` が `yomitoku` のときに使用される |

`ocr.yomitoku_device` (既定 `mps`) と `ocr.prewarm_yomitoku` を参照します。
結果が空の場合は parser が tesseract にフォールバックします。

**役割**

日本語抽出精度を優先した OCR 経路を提供します。

---

### 3.5 pytesseract

- **パッケージ**: `pytesseract>=0.3.13`
- **公式サイト**: https://github.com/madmaze/pytesseract

**概要**

Google の Tesseract OCR エンジンの Python ラッパーです。
実行には Tesseract バイナリの別途インストールが必要です。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | `ocr.engine="tesseract"` での primary OCR、および `easyocr` / `yomitoku` の fallback として使用される |

`ocr.config` (`--oem 3 --psm 6` 等) および `ocr.default_lang` / `ocr.japanese_lang` / `ocr.fallback_lang` の設定を参照します。

**役割**

EasyOCR の代替として、軽量な Tesseract ベースの OCR を提供します。

---

### 3.6 OpenCV

- **パッケージ**: `opencv-python>=4.8.0`
- **公式サイト**: https://opencv.org

**概要**

コンピュータービジョン用の包括的ライブラリです。
画像の前処理・変換・分析に広く利用されています。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | OCR 前処理として、画像のグレースケール変換・二値化・ノイズ除去などを行う |

**役割**

OCR エンジンに渡す前の画像品質を向上させ、OCR 精度を高めます。

---

### 3.7 Pillow

- **パッケージ**: `pillow>=12.1.1`
- **公式サイト**: https://pillow.readthedocs.io
- **インポート名**: `from PIL import Image`

**概要**

Python の標準的な画像処理ライブラリ (PIL の後継) です。
多様な画像フォーマットの読み書きと基本的な画像操作をサポートします。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/core/parser.py` | PyMuPDF がラスタライズしたページ画像を PIL Image として扱う。`RawChunk.raw_content` に PIL Image を格納する |
| `src/agents/extraction.py` | `VisionAgent` が PIL Image をバイト列に変換して Ollama の vision API に送信する |
| `src/agents/validation.py` | `ChunkValidatorAgent` が figure チャンクの PIL Image を VLM に送信する |

**役割**

PDF ページ画像・図チャンクをパイプライン全体で統一的に扱うための画像データ表現として機能します。

---

## 4. オブザーバビリティ

### 4.1 Langfuse

- **パッケージ**: `langfuse>=3.14.4`
- **公式サイト**: https://langfuse.com

**概要**

LLM アプリケーション向けのオブザーバビリティプラットフォームです。
トレース・スパン・スコアを記録し、LLM の動作をデバッグ・モニタリングできます。
セルフホストとクラウドの両方に対応しています。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/integrations/langfuse.py` | `LangfuseTracer` がパイプラインの各フェーズをトレース / スパン / ジェネレーションとして記録する |

記録される主なトレース:

```
ingest_pdf トレース
  ├── Span: parse_pdf
  ├── Span: agent_text (× N)
  ├── Span: agent_table (× M)
  ├── Span: agent_vision (× K)
  └── Span: upsert_store

rag_query トレース
  ├── Span: retrieve_chunks
  ├── Span: retrieve_figures (条件付き)
  └── Generation: orchestrator_reasoning
        input: プロンプト
        output: 回答
        model: <model_id>
        usage: トークン数
```

スコアも記録します:
- `chunk_quality`: チャンク抽出品質スコア (CHECKPOINT A)
- `answer_grounding`: 回答グラウンディングスコア (CHECKPOINT B)

環境変数 (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`) が未設定の場合は no-op で動作します。

**役割**

LLM の推論コスト・品質・ハルシネーション率などをモニタリングするためのオブザーバビリティ基盤です。
開発者が推論の内部動作を可視化し、モデル選択やプロンプト改善に役立てることができます。

---

## 5. データモデル / ユーティリティ

### 5.1 Pydantic

- **パッケージ**: `pydantic>=2.12.5`
- **公式サイト**: https://docs.pydantic.dev

**概要**

Python の型アノテーションを使ったデータバリデーション・シリアライゼーションライブラリです。
`BaseModel` を継承したクラスで型安全なデータ構造を定義できます。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/integrations/dspy_modules.py` | `AnswerGroundingOutput` として DSPy の構造化出力スキーマを定義する。`Field` を使ってフィールドの説明・バリデーション条件 (`ge=0.0`, `le=1.0` 等) を記述する |

**役割**

DSPy が LLM から受け取る非構造化テキストレスポンスを、型安全な Python オブジェクトに変換します。
`verdict_score` の範囲制約など、LLM 出力の品質チェックもバリデーション層で担保します。

---

### 5.2 tiktoken

- **パッケージ**: `tiktoken>=0.12.0`
- **公式サイト**: https://github.com/openai/tiktoken

**概要**

OpenAI が開発した高速 BPE トークナイザーです。
GPT 系モデルのトークン計算に使われますが、他モデルの概算にも利用できます。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/utils/token_counter.py` | `count_tokens()` が `cl100k_base` エンコーディングでテキストのトークン数を近似計算し、Langfuse の `usage_details` に渡す |

モデル固有のトークナイザーが利用できない場合のフォールバックとして動作します。
日本語 CJK テキストへの最終フォールバックは文字数ベースの推定 (`len(text) // 2`) です。

**役割**

Langfuse へ送信する推論コスト (トークン使用量) を、モデルに依存しない近似値で算出します。

---

### 5.3 python-dotenv

- **パッケージ**: `python-dotenv>=1.2.1`
- **公式サイト**: https://github.com/theskumar/python-dotenv

**概要**

`.env` ファイルから環境変数を読み込むユーティリティです。
`load_dotenv()` を呼び出すだけで `.env` ファイルの変数が `os.environ` に追加されます。

**このアプリでの用途**

| モジュール | 用途 |
|-----------|------|
| `src/integrations/langfuse.py` | `LangfuseTracer` の初期化時に `load_dotenv()` を呼び出し、`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` などの認証情報を `.env` から読み込む |

**役割**

Langfuse の認証情報をソースコードに埋め込まずに管理するための仕組みです。
ローカル開発では `.env` ファイル、本番環境では OS の環境変数を使うというパターンを実現します。

---

## 補足: grpcio

- **パッケージ**: `grpcio!=1.78.1`

ChromaDB の内部通信プロトコルとして gRPC を使用するため、間接依存として含まれています。
バージョン `1.78.1` は既知の問題があるため除外されています。
アプリコードから直接使用することはありません。
