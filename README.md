# Agentic RAG for Multi-Model PDF Extraction

学術論文や政府文書などの複雑なPDFから、**テキスト・テーブル・図表**を自動的に抽出し、マルチモーダルRAGパイプラインで高精度な質問応答を実現する、Apple Silicon最適化のエージェント型システムです。

## ✨ 主要機能

### 🎯 マルチモーダルPDF解析
- **自動チャンク分類**: テキスト、テーブル、図表を自動認識
- **専用エージェント処理**: 各チャンクタイプに特化した小型言語モデル（SLM）で最適化
- **自己リフレクション**: 信頼度スコア < 0.5 の場合、自動的に再試行

### 🛡️ 2段階バリデーション
- **CHECKPOINT A（ChunkValidator）**: 抽出直後のチャンク品質監査
  - 原文との整合性チェック
  - 図表は画像を直接検証
  - 不正なチャンクは修正または破棄
- **CHECKPOINT B（AnswerValidator）**: RAG回答の幻覚検出
  - 回答の各主張がソースに基づいているか検証
  - 根拠のない主張を検出して修正

### ⚡ メモリ効率的なモデル管理
- **Sequential Loading**: 大型モデル（8B）は必要時のみロード/アンロード
- **4-bit量子化**: MLX最適化により、VRAMピーク使用量は**4-5GB**
- **Apple Silicon対応**: M1/M2/M3チップで高速動作

### 🔍 高精度なRAG検索
- セマンティック検索（多言語対応）
- ビジュアルキーワード検出時の図表優先検索
- スコアベースのランキングとソース引用

## 🚀 クイックスタート

### 前提条件
- **Python**: 3.13以上
- **OS**: macOS（Apple Silicon推奨）
- **メモリ**: 8GB以上のRAM（16GB推奨）

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd agentic_rag_for_multi_model_pdf_extraction

# uv（推奨）を使用してインストール
uv sync

# または、pipを使用
pip install -e .
```

### 環境変数の設定

`.env`ファイルをプロジェクトルートに作成：

```bash
# HuggingFace認証（モデルダウンロード用）
HF_TOKEN=your_huggingface_token

# Langfuseトレーシング（オプション・現在無効化中）
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# モデルキャッシュディレクトリ（オプション）
HF_HOME=./models
```

### 基本的な使い方

```bash
# PDFを処理してチャンクを抽出・保存
uv run ./agentic_rag_flow.py ./input/your_paper.pdf

# バリデーションを有効化して処理
uv run ./agentic_rag_flow.py ./input/your_paper.pdf --validate

# RAGクエリを実行
uv run ./agentic_rag_flow.py ./input/your_paper.pdf --query "図2は何を示していますか？"
```

### 出力ファイル

```
output/
├── your_paper_chunks.json    # 抽出されたチャンク（構造化テキスト、概念、信頼度）
└── your_paper_answer.json    # RAG回答（検証結果、ソース引用、推論過程）
```

## 🏗️ アーキテクチャ

### データフロー

```
PDF入力
  ↓
[PDFParser] pymupdf + pdfplumber
  ↓
RawChunk (TEXT | TABLE | FIGURE)
  ↓
┌─────────────┬─────────────┬─────────────┐
│ TextAgent   │ TableAgent  │ VisionAgent │
│ (MLX 4B)    │ (MLX 3B)    │ (MLX 256M)  │
└─────────────┴─────────────┴─────────────┘
  ↓
ProcessedChunk (structured_text, key_concepts, confidence)
  ↓
[ChunkValidator] CHECKPOINT A (MLX 256M)
  ↓
[ChromaDB Store] e5-small-multilingual embeddings
  ↓
[ReasoningOrchestrator] RAG検索 + 回答生成 (MLX 8B)
  ↓
[AnswerValidator] CHECKPOINT B (MLX 8B)
  ↓
RAGAnswer (validation, sources, reasoning)
```

### 使用モデル（MLX最適化・4-bit量子化）

| 役割 | モデル | サイズ | メモリ | 用途 |
|------|--------|--------|--------|------|
| **Text抽出** | mlx-community/Phi-3.5-mini-Instruct-4bit | 3.8B | ~2GB | テキスト正規化・概念抽出 |
| **Table抽出** | mlx-community/Qwen2.5-3B-Instruct-4bit | 3B | ~1.5GB | テーブルスキーマ推論・修復 |
| **Vision抽出** | mlx-community/SmolVLM-256M-Instruct-4bit | 256M | ~1GB | 図表分類・説明生成 |
| **Orchestrator** | mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit | 8B | ~4GB | RAG推論・回答生成（CoT） |
| **Chunk検証** | mlx-community/SmolVLM-256M-Instruct-4bit | 256M | ~1GB | チャンク品質監査（CHECKPOINT A） |
| **Answer検証** | mlx-community/Qwen3-8B-4bit | 8B | ~4GB | 幻覚検出（CHECKPOINT B） |
| **Embedder** | intfloat/multilingual-e5-small | 118M | ~500MB | ベクトル化（多言語対応） |

**VRAMピーク**: 約4-5GB（Sequential loading使用時）

## 📁 ディレクトリ構造

```
agentic_rag_for_multi_model_pdf_extraction/
├── agentic_rag_flow.py      # メインパイプライン（1172行）
├── validator_agent.py        # バリデーションエージェント（473行）
├── langfuse_tracer.py        # トレーシング（現在無効化）
├── pyproject.toml            # 依存関係定義
├── README.md                 # このファイル
├── ARCHITECTURE.md           # 技術詳細ドキュメント
├── .env                      # 環境変数（要作成）
├── .gitignore                # Git除外設定
│
├── input/                    # 処理対象のPDFファイルを配置
├── output/                   # 処理結果（チャンク、回答）
├── chroma_db/                # ベクトルDB永続化
├── models/                   # ローカルモデルキャッシュ
└── attics/                   # 旧バージョン情報
```

## 🔧 技術スタック

### コア依存関係

```toml
[dependencies]
# ML/AI
mlx>=0.1.0                    # Apple Silicon最適化
mlx-lm>=0.1.0                 # MLX言語モデル
mlx-vlm>=0.1.0                # MLX Vision-Language モデル
sentence-transformers>=5.2.3  # 埋め込みモデル

# PDF処理
pymupdf>=1.27.1               # 画像・テキスト抽出
pdfplumber>=0.11.9            # テーブル抽出
pillow>=12.1.1                # 画像処理
pytesseract>=0.3.13           # OCRフォールバック

# ベクトルDB・トレーシング
chromadb>=1.5.1               # ベクトルストア
langfuse>=3.14.4              # 観測可能性（現在無効）

# その他
python-dotenv>=1.2.1          # 環境変数管理
unstructured>=0.20.8          # ドキュメント処理ユーティリティ
```

## 🎓 主要クラス

### `AgenticRAGPipeline`
メインのパイプラインクラス。PDF処理、チャンク抽出、ベクトルストアへの保存、RAGクエリを統合。

```python
rag = AgenticRAGPipeline(
    pdf_parser=PDFParser(),
    router=AgentRouter(...),
    chunk_store=ChunkStore(...),
    orchestrator=ReasoningOrchestratorAgent(...),
    validator=ChunkValidatorAgent(...),
    answer_validator=AnswerValidatorAgent(...),
)

# チャンク抽出とバリデーション
chunks = rag.ingest(pdf_path, validates=True)

# RAGクエリ
answer = rag.query(question, validates=True)
```

### `BaseLoadableModel`
モデルのライフサイクル管理Mixin。明示的なload/unloadでメモリ効率化。

```python
# コンテキストマネージャーとして使用
with orchestrator:
    answer = orchestrator.generate(query, context)
# ブロック終了時に自動的にモデルをアンロード
```

### `ChunkValidatorAgent` / `AnswerValidatorAgent`
2段階バリデーションを実装。チャンク品質監査と回答幻覚検出。

## 🐛 トラブルシューティング

### モデルのダウンロードが失敗する

```bash
# HuggingFace認証情報を設定
export HF_TOKEN=your_token_here

# または、.envファイルに記載
echo "HF_TOKEN=your_token_here" >> .env
```

### メモリ不足エラー

- 小型モデル（2-4B）は常駐しますが、大型モデル（8B）はSequential loadingで管理されています
- バリデーションを無効化（`validates=False`）すると、メモリ使用量が削減されます
- macOSのアクティビティモニタで実際のメモリ使用量を確認してください

### テーブル抽出精度が低い

- スキャンPDFの場合、pdfplumberは境界線のないテーブルに弱い
- Camelot（latticeモード）やAWS Textractの使用を検討してください
- デジタルネイティブPDFでも、複雑な結合セルは誤検出されることがあります

### Langfuseトレーシングが動作しない

- **現在、Langfuse SDK API非互換により無効化されています**
- トレーシング機能は`no-op`として動作し、処理には影響しません
- 将来のSDKアップデートで再有効化される予定です

## 📚 詳細ドキュメント

より詳細な技術仕様、設計判断、メモリ管理戦略については、[ARCHITECTURE.md](ARCHITECTURE.md)を参照してください。

## 🤝 貢献

プルリクエスト、イシュー報告、機能提案を歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Note**: このプロジェクトはApple Silicon（M1/M2/M3）向けに最適化されており、MLXライブラリを使用しています。Intel MacやLinux/Windowsでは、transformersライブラリへの移行が必要な場合があります。
