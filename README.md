# Agentic RAG for Multi-Model PDF Extraction

学術論文や政府文書などの複雑なPDFから、**テキスト・テーブル・図表**を自動的に抽出し、マルチモーダルRAGパイプラインで高精度な質問応答を実現する、Apple Silicon最適化のエージェント型システムです。

## ✨ 主要機能

### 🎯 マルチモーダルPDF解析
- **自動チャンク分類**: テキスト、テーブル、図表を自動認識
- **専用エージェント処理**: 各チャンクタイプに特化した小型言語モデル（SLM）で最適化
- **自己リフレクション**: 信頼度スコア < 0.5 の場合、自動的に再試行

### 🛡️ 2段階バリデーション（DSPy強化）
- **CHECKPOINT A（ChunkValidator）**: 抽出直後のチャンク品質監査
  - 原文との整合性チェック
  - 図表は画像を直接検証
  - 不正なチャンクは修正または破棄
- **CHECKPOINT B（AnswerValidator）**: RAG回答の幻覚検出（**DSPy統合済み**）
  - 回答の各主張がソースに基づいているか検証
  - 根拠のない主張を検出して修正
  - `ChainOfThought`による段階的検証
  - 構造化出力で精密なハルシネーション特定

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
# PDFインジェスト（チャンク抽出・保存）
python app.py ingest ./input/your_paper.pdf

# バリデーション付きインジェスト
python app.py ingest ./input/your_paper.pdf --validate

# RAGクエリ実行
python app.py query "図2は何を示していますか？"

# フルパイプライン（インジェスト + クエリ）
python app.py pipeline ./input/your_paper.pdf "図2は何を示していますか？" --validate

# ヘルプを表示
python app.py --help
python app.py ingest --help
python app.py query --help
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
├── app.py                    # CLIエントリーポイント（540行）
├── agentic_rag_flow.py      # 後方互換ラッパー（非推奨）
├── pyproject.toml            # 依存関係・パッケージ定義
├── README.md                 # このファイル
├── ARCHITECTURE.md           # 技術詳細ドキュメント
├── MIGRATION.md              # v0.3.0移行ガイド
├── .env                      # 環境変数（要作成）
├── .gitignore                # Git除外設定
│
├── src/                      # メインパッケージ
│   ├── core/                 # コア機能
│   │   ├── models.py         # データ構造（ChunkType, RAGAnswerなど）
│   │   ├── cache.py          # モデルキャッシュ管理
│   │   ├── parser.py         # PDFParser（pdfplumber + PyMuPDF）
│   │   ├── store.py          # ChromaDB ベクトルストア
│   │   └── pipeline.py       # AgenticRAGPipeline メイン実装
│   ├── agents/               # AIエージェント
│   │   ├── base.py           # BaseAgent, BaseLoadableModel
│   │   ├── extraction.py     # Text/Table/Visionエージェント
│   │   ├── router.py         # AgentRouter（チャンク振り分け）
│   │   ├── orchestrator.py   # ReasoningOrchestratorAgent（RAG推論）
│   │   └── validation.py     # Chunk/Answer バリデーター
│   ├── integrations/         # 外部統合
│   │   ├── dspy_modules.py   # DSPy Signatures & Pydantic モデル
│   │   ├── dspy_adapter.py   # MLXLM（DSPy ⇔ MLX ブリッジ）
│   │   └── langfuse.py       # LangfuseTracer（オブザーバビリティ）
│   └── utils/                # ユーティリティ
│       └── serialization.py  # JSON出力ヘルパー
│
├── tests/                    # pytestテストスイート
│   ├── conftest.py           # 共通フィクスチャ
│   ├── test_models.py        # データ構造のユニットテスト
│   ├── test_pipeline.py      # パイプライン統合テスト
│   └── test_dspy_validator.py # DSPy検証テスト
│
├── input/                    # 処理対象のPDFファイルを配置
├── output/                   # 処理結果（チャンク、回答）
├── chroma_db/                # ベクトルDB永続化
├── models/                   # ローカルモデルキャッシュ
└── attics/                   # 旧バージョン情報・ドキュメント
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
dspy-ai>=2.5.0                # プロンプト最適化フレームワーク（PHASE 2）

# PDF処理
pymupdf>=1.27.1               # 画像・テキスト抽出
pdfplumber>=0.11.9            # テーブル抽出
pillow>=12.1.1                # 画像処理
pytesseract>=0.3.13           # OCRフォールバック

# ベクトルDB・トレーシング
chromadb>=1.5.1               # ベクトルストア
langfuse>=3.14.4              # 観測可能性・トレーシング（PHASE 1）

# その他
python-dotenv>=1.2.1          # 環境変数管理
unstructured>=0.20.8          # ドキュメント処理ユーティリティ
```

## 🎓 主要クラス

### `AgenticRAGPipeline`
メインのパイプラインクラス。PDF処理、チャンク抽出、ベクトルストアへの保存、RAGクエリを統合。

```python
from src.core.pipeline import AgenticRAGPipeline
from src.core.parser import PDFParser
from src.agents.router import AgentRouter
from src.core.store import ChunkStore

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

**Note**: v0.3.0では、すべてのコアクラスが`src/`パッケージに移行しました。後方互換性のため、`from agentic_rag_flow import ...`もまだ動作しますが、非推奨警告が表示されます。詳細は[MIGRATION.md](MIGRATION.md)を参照してください。

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

## 🧪 DSPy統合（PHASE 2完了）

### 概要

AnswerValidatorAgentに**DSPy（Declarative Self-improving Language Programs）**を統合し、プロンプト最適化と構造化出力を実現しました。

### 実装状況

- ✅ **完了**: AnswerValidatorAgent（2026-02-23）
- ⏳ **未実装**: ChunkValidatorAgent、その他エージェント

### DSPyモードの効果

従来のレガシーモードと比較して、以下の改善が確認されています：

| 項目 | Legacy | DSPy | 改善率 |
|------|--------|------|--------|
| ハルシネーション検出精度 | 文全体を一括判定 | 節レベルで特定 | ✅ 精密化 |
| 部分的正解のスコアリング | 0.00（失敗扱い） | 0.20（認識） | ✅ +20pt |
| 出力パース | 正規表現（脆弱） | Pydantic（型安全） | ✅ 堅牢化 |
| 推論の可視性 | なし | ChainOfThought | ✅ トレース可能 |

### 使用方法

```python
from src.agents.validation import AnswerValidatorAgent

# DSPyモードで使用（デフォルト）
answer_validator = AnswerValidatorAgent(
    model_loader=answer_validator_model,
    use_dspy=True  # デフォルトで有効
)

# レガシーモードに戻す（比較用）
answer_validator = AnswerValidatorAgent(
    model_loader=answer_validator_model,
    use_dspy=False
)
```

### アーキテクチャ

DSPy統合には以下のコンポーネントが含まれます：

- **[src/integrations/dspy_adapter.py](src/integrations/dspy_adapter.py)**: DSPyフレームワークとMLXモデルの橋渡し（`MLXLM`クラス）
- **[src/integrations/dspy_modules.py](src/integrations/dspy_modules.py)**: DSPy SignaturesとPydanticモデル
  - `AnswerGroundingSignature`: 幻覚検出タスクのDSPy署名
  - `AnswerGroundingOutput`: 構造化出力用Pydanticモデル
- **`dspy.ChainOfThought`**: 段階的推論モジュール（DSPyフレームワーク組み込み）

### 今後の計画

- ChunkValidatorAgentへの適用（中優先度）
- DSPy optimizers（BootstrapFewShot, MIPRO）の導入
- 本番Langfuseメトリクスでの効果測定

詳細は[PLAN.md](PLAN.md)を参照してください。

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

### Langfuseトレーシング

- Langfuse SDK v3.14.4と完全に統合されています
- 環境変数`LANGFUSE_PUBLIC_KEY`と`LANGFUSE_SECRET_KEY`を設定すると、自動的にトレースが記録されます
- 詳細な実装情報については[attics/PHASE_1_LANGFUSE_FIXES.md](attics/PHASE_1_LANGFUSE_FIXES.md)を参照してください

## 📚 詳細ドキュメント

より詳細な技術仕様、設計判断、メモリ管理戦略については、[ARCHITECTURE.md](ARCHITECTURE.md)を参照してください。

## 🤝 貢献

プルリクエスト、イシュー報告、機能提案を歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Note**: このプロジェクトはApple Silicon（M1/M2/M3）向けに最適化されており、MLXライブラリを使用しています。Intel MacやLinux/Windowsでは、transformersライブラリへの移行が必要な場合があります。
