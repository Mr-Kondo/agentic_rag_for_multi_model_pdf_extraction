# Agentic RAG Flow — Architecture (v3)

> **Version**: 3.0  
> **Last Updated**: 2026-02-20  
> **Apple Silicon対応**: MLX最適化版

---

## 🎯 v3の主要な変更点

### v2からの移行

| 項目 | v2 | v3 |
|------|----|----|
| **MLライブラリ** | transformers + torch | **MLX** (Apple Silicon最適化) |
| **量子化** | 8-bit (bitsandbytes) | **4-bit** (MLX native) |
| **メモリ管理** | 3つの小型SLMを同時ロード | **BaseLoadableModel**パターン<br>Sequential loading戦略 |
| **VRAMピーク** | ~16-22GB (FP16) | **~4-5GB** (4-bit quantized) |
| **Orchestrator** | Phi-3.5-mini (3.8B) | **DeepSeek-R1-Distill-Llama-8B** (8B)<br>推論能力強化、CoT出力 |
| **Validator構成** | 単一ValidatorAgent | **2つの専用バリデーター**:<br>• ChunkValidatorAgent<br>• AnswerValidatorAgent |
| **Vision検証** | テキストベースのみ | **画像を直接検証**（VLMモデル） |
| **トレーシング** | Langfuse完全実装 | ✅ **完全動作**（v3.14.4対応） |

---

## 📐 システムアーキテクチャ

### データフローダイアグラム

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PDF INPUT (学術論文/政府文書)                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                   ┌─────────▼─────────┐
                   │    PDFParser      │  pymupdf + pdfplumber
                   │                   │  → RawChunk(TEXT | TABLE | FIGURE)
                   └─────────┬─────────┘
                             │
                ┌────────────▼────────────┐
                │     AgentRouter         │  チャンクタイプによるルーティング
                └──┬─────────────┬────┬──┘
                   │             │    │
          ┌────────▼───┐  ┌─────▼────┐  ┌────▼──────┐
          │ TextAgent  │  │TableAgent│  │VisionAgent│
          │ (MLX 4B)   │  │(MLX 3B)  │  │(MLX 256M) │
          │Phi-3.5-mini│  │Qwen2.5-3B│  │SmolVLM-256│
          └────────┬───┘  └─────┬────┘  └────┬──────┘
                   │             │            │
                   │   Self-Reflection Loop   │
                   │   (retry if conf < 0.5)  │
                   │             │            │
          ┌────────▼─────────────▼────────────▼──────┐
          │          ProcessedChunk                  │
          │  structured_text | intuition_summary     │
          │  key_concepts | confidence | agent_notes │
          └────────────────────┬──────────────────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │  ChunkValidatorAgent (CHECKPOINT A)│  load/unload
             │  (MLX SmolVLM-256M)                │  画像も直接検証
             │  → ChunkValidationResult           │
             │     • is_valid                     │
             │     • corrected (ProcessedChunk)   │
             │     • verdict_score                │
             └─────────────────┬──────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   ChromaDB Store  │  e5-small-multilingual
                     │   (vector store)  │  多言語埋め込み
                     └─────────┬─────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │  ReasoningOrchestratorAgent        │
             │  .retrieve() → hits (no model)     │  セマンティック検索
             │  .generate() → answer (load/unload)│  load/unload
             │  (MLX DeepSeek-R1-8B)              │  CoT推論
             └─────────────────┬──────────────────┘
                               │
             ┌─────────────────▼──────────────────┐
             │ AnswerValidatorAgent (CHECKPOINT B)│  load/unload
             │ (MLX Qwen3-8B)                     │  幻覚検出
             │ → AnswerValidationResult           │
             │     • is_grounded                  │
             │     • corrected_answer             │
             └─────────────────┬──────────────────┘
                               │
                        ┌──────▼───────┐
                        │  RAGAnswer   │
                        │  + validation│
                        │  + sources   │
                        │  + reasoning │
                        └──────────────┘
```

---

## 🤖 エージェント設計

### 3つの専用抽出エージェント

| Agent | 入力 | MLXモデル | 責務 | 主要出力フィールド |
|-------|------|----------|------|-------------------|
| **TextAgent** | 生テキスト | Phi-3.5-mini-4bit (3.8B) | ハイフン除去、セクション正規化、概念抽出 | `structured_text`, `key_concepts` |
| **TableAgent** | Markdownテーブル | Qwen2.5-3B-4bit (3B) | スキーマ推論、単位抽出、結合セル修復 | `structured_text`, `schema` |
| **VisionAgent** | PIL.Image | SmolVLM-256M-4bit (256M) | 図表分類、軸読み取り、フロー説明 | `figure_type`, `intuition_summary` |

### Self-Reflection機構

各エージェントは自身の`confidence`スコア（0–1）をJSONレスポンスで返します。

```python
if confidence < 0.5:
    # 厳格化されたプロンプトで再実行
    result = agent.retry_with_strict_prompt(chunk)
```

これは、別個の批評モデルの軽量な代替手段です。

---

## 🔍 使用モデル一覧（v3）

### MLX最適化モデル（4-bit量子化）

| 役割 | HuggingFace モデルID | サイズ | VRAM | 特徴 |
|------|---------------------|--------|------|------|
| **Text抽出** | `mlx-community/Phi-3.5-mini-Instruct-4bit` | 3.8B | ~2GB | テキスト正規化、概念抽出に特化 |
| **Table抽出** | `mlx-community/Qwen2.5-3B-Instruct-4bit` | 3B | ~1.5GB | 構造化データ理解、スキーマ推論 |
| **Vision抽出** | `mlx-community/SmolVLM-256M-Instruct-4bit` | 256M | ~1GB | 超軽量VLM、図表分類・説明生成 |
| **Orchestrator** | `mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit` | 8B | ~4GB | **推論能力強化**、CoT出力（`<think>`ブロック） |
| **Chunk検証** | `mlx-community/SmolVLM-256M-Instruct-4bit` | 256M | ~1GB | CHECKPOINT A: チャンク品質監査、画像検証 |
| **Answer検証** | `mlx-community/Qwen3-8B-4bit` | 8B | ~4GB | CHECKPOINT B: 幻覚検出、根拠検証 |
| **Embedder** | `intfloat/multilingual-e5-small` | 118M | ~500MB | セマンティック埋め込み、多言語対応 |

**総VRAMピーク（Sequential loading時）**: ~4-5GB

---

## 💾 メモリ管理戦略（v3の核心機能）

### BaseLoadableModelパターン

v3では、大型モデル（8B）のメモリ効率化のため、明示的なライフサイクル管理を実装しています。

```python
class BaseLoadableModel:
    """
    明示的なload/unloadライフサイクルを持つモデルMixin。
    コンテキストマネージャーとして使用可能。
    """
    
    def load(self) -> None:
        """モデルをメモリにロード"""
        
    def unload(self) -> None:
        """モデルをメモリから解放（ガベージコレクション実行）"""
        
    def __enter__(self):
        self.load()
        return self
        
    def __exit__(self, *args):
        self.unload()
```

### Sequential Loading戦略

**原則**: 大型モデル（8B以上）は**一度に1つのみ**メモリに配置。

| フェーズ | ロード中のモデル | メモリ使用量 |
|---------|----------------|------------|
| **抽出フェーズ** | TextAgent (2GB) + TableAgent (1.5GB) + VisionAgent (1GB) | ~4.5GB |
| **検証フェーズ（A）** | ChunkValidator (1GB) のみ | ~1GB |
| **RAG検索フェーズ** | なし（埋め込みモデルのみ、500MB） | ~500MB |
| **回答生成フェーズ** | Orchestrator (4GB) のみ | ~4GB |
| **検証フェーズ（B）** | AnswerValidator (4GB) のみ | ~4GB |

**使用例**:

```python
# Orchestratorは必要時のみロード
with orchestrator:  # __enter__でload()
    answer = orchestrator.generate(query, context)
# __exit__でunload() → ガベージコレクション

# AnswerValidatorも同様
with answer_validator:
    validation = answer_validator.validate(answer, sources)
```

### ModelCacheシステム

```python
class ModelCache:
    """
    HuggingFace Hub経由でMLXモデルをダウンロード・キャッシュ。
    環境変数HF_HOMEで指定されたディレクトリに保存。
    """
    
    @classmethod
    def get_model_path(cls, model_id: str) -> str:
        """ローカルキャッシュパスを返すか、ダウンロードを実行"""
```

---

## 🛡️ 2段階バリデーション（CHECKPOINT A & B）

### CHECKPOINT A: ChunkValidator

**タイミング**: チャンク抽出直後、ベクトルストアに保存する前

**目的**: 抽出品質の監査、不正なチャンクの修正または破棄

```python
@dataclass
class ChunkValidationResult:
    is_valid: bool                         # バリデーション結果
    issues: list[str]                      # 検出された問題のリスト
    corrected: ProcessedChunk | None       # 修正されたチャンク（不正な場合）
    verdict_score: float                   # 品質スコア（0-1）
    validator_notes: str                   # バリデーターのコメント
```

**検証内容**:
- 原文との整合性チェック
- 図表の場合、**画像を直接VLMで検証**
- 構造化テキストの完全性
- 重要概念の欠落検出

**処理フロー**:

```python
for chunk in extracted_chunks:
    with chunk_validator:  # load/unload
        validation = chunk_validator.validate(chunk, raw_chunk)
        
    if not validation.is_valid:
        if validation.corrected:
            chunk = validation.corrected  # 修正版を使用
        else:
            continue  # チャンクを破棄
            
    chunk_store.add(chunk)
```

### CHECKPOINT B: AnswerValidator

**タイミング**: RAG回答生成後、ユーザーに返す前

**目的**: 幻覚検出、根拠のない主張の修正

```python
@dataclass
class AnswerValidationResult:
    is_grounded: bool                      # 回答が根拠に基づいているか
    ungrounded_claims: list[str]           # 根拠のない主張のリスト
    corrected_answer: str | None           # 修正された回答
    grounding_score: float                 # 根拠スコア（0-1）
    validator_notes: str                   # バリデーターのコメント
```

**検証内容**:
- 回答の各主張がソースチャンクに基づいているか検証
- 過度の一般化や捏造の検出
- ソース引用の正確性チェック

**処理フロー**:

```python
with answer_validator:  # load/unload
    validation = answer_validator.validate(answer, source_chunks)
    
if not validation.is_grounded:
    if validation.corrected_answer:
        answer.text = validation.corrected_answer  # 修正版を使用
    else:
        answer.add_warning("回答に根拠のない主張が含まれています")
```

---

## 🔎 検索戦略

### セマンティック検索（ChromaDB + e5-small）

1. **標準検索**（全チャンクタイプ）
   - トップ8結果を取得
   - コサイン類似度でランキング
   
2. **ビジュアルキーワード検出時の図表優先検索**
   - クエリに視覚的キーワード（"図", "グラフ", "フロー", "図表"等）が含まれる場合
   - 追加でFIGUREタイプのみのトップ3検索を実行
   - 重複排除してマージ

3. **コンテキストウィンドウ**
   - Orchestratorへのコンテキスト: 8–11チャンク × 800文字/チャンク ≈ 6,400–8,800トークン
   
**検索実装例**:

```python
def retrieve(self, query: str, top_k: int = 8) -> list[ProcessedChunk]:
    # 標準検索
    results = self.chunk_store.query(query, n_results=top_k)
    
    # ビジュアルキーワード検出
    visual_keywords = ["図", "グラフ", "フロー", "figure", "graph", "chart"]
    if any(kw in query.lower() for kw in visual_keywords):
        # 図表専用検索を追加
        figure_results = self.chunk_store.query(
            query, 
            n_results=3,
            where={"chunk_type": "FIGURE"}
        )
        # 重複排除してマージ
        results = self._deduplicate_and_merge(results, figure_results)
    
    return results
```

---

## 🔄 モデルスワップガイド

MLXエコシステム内でのモデル交換は容易です。

| スロット | デフォルト（v3） | 軽量版 | 重量版 |
|----------|----------------|--------|--------|
| **Text SLM** | Phi-3.5-mini-4bit (3.8B) | Qwen2.5-1.5B-Instruct-4bit | Mistral-7B-Instruct-4bit |
| **Table SLM** | Qwen2.5-3B-4bit | Phi-3.5-mini-4bit | Qwen2.5-7B-Instruct-4bit |
| **Vision SLM** | SmolVLM-256M-4bit | （最軽量版） | Phi-3.5-vision-4bit (4.2B) |
| **Orchestrator** | DeepSeek-R1-8B-4bit | Phi-3.5-mini-4bit | DeepSeek-R1-14B-4bit |
| **Embedder** | e5-small-multilingual | paraphrase-multilingual-mpnet | OpenAI text-embedding-3-small (API) |

**注意**: 
- Intel MacやLinux/Windowsでは、MLXモデルは動作しません
- その場合、`transformers`ライブラリとPyTorch版モデルへの移行が必要です

---

## ⚠️ 制限事項と考慮事項

### 1. テーブル検出精度

**問題**: pdfplumberは境界線のないテーブルや結合セルを持つスキャンPDFに弱い。

**回避策**:
- デジタルネイティブPDFを使用
- Camelot（latticeモード）やAWS Textractを検討
- テーブル検出前処理として、OpenCVベースの境界線検出を追加

### 2. Vision SLMの幻覚

**問題**: 256Mクラスのモデルは、レイアウトは正確に説明できるが、軸の値や数値データを誤読することがある。

**回避策**:
- 図表に数値グリッドが含まれる場合、TableAgentと併用
- 重要な数値データはOCR（pytesseract）で補完
- より大型のVLM（Phi-3.5-vision 4.2B）にアップグレード

### 3. OCRは自動起動しない

**問題**: このパイプラインはデジタルネイティブPDFを前提としており、スキャンPDFには対応していない。

**回避策**:
- スキャンPDFの場合、前処理でOCR（pytesseract, EasyOCR, AWS Textract）を実行
- または、`unstructured[all-docs]`の自動OCR機能を活用

### 4. 信頼度スコアは自己評価

**問題**: SLMは自身の出力を評価するため、敵対的入力や分布外入力には信頼性が低い。

**回避策**:
- 外部スコアリングモデルを追加
- 人間の監査ループを組み込む
- CHECKPOINT A/Bバリデーションで二重チェック

### 5. Langfuseトレーシングの実装状態

**ステータス**: ✅ **完全動作中**（Langfuse SDK v3.14.4対応）

**実装内容**:
- Phase 1で完全に修正・統合済み（[詳細](attics/PHASE_1_LANGFUSE_FIXES.md)）
- トレース、スパン、ジェネレーション、スコアリングAPIがすべて正常動作
- トークンカウントも正確に記録

**使用方法**:
```bash
# .envファイルで環境変数を設定
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

環境変数が未設定の場合、トレーシングは自動的にスキップされ、処理には影響しません。

---

## 📦 インストール

### 前提条件

- **Python**: 3.13以上
- **OS**: macOS（Apple Silicon: M1/M2/M3推奨）
- **メモリ**: 最低8GB RAM（16GB推奨）
- **ストレージ**: 約20GB（モデルキャッシュ用）

### 依存関係のインストール

```bash
# uvを使用（推奨）
uv sync

# または、pipを使用
pip install -e .
```

### 主要依存関係（pyproject.toml）

```toml
[project]
dependencies = [
    "mlx>=0.1.0",                    # Apple Silicon最適化フレームワーク
    "mlx-lm>=0.1.0",                 # MLX言語モデル
    "mlx-vlm>=0.1.0",                # MLX Vision-Languageモデル
    "sentence-transformers>=5.2.3",  # 埋め込みモデル
    "chromadb>=1.5.1",               # ベクトルストア
    "pymupdf>=1.27.1",               # PDF解析
    "pdfplumber>=0.11.9",            # テーブル抽出
    "pillow>=12.1.1",                # 画像処理
    "pytesseract>=0.3.13",           # OCRフォールバック
    "langfuse>=3.14.4",              # トレーシング（現在無効）
    "python-dotenv>=1.2.1",          # 環境変数管理
    "unstructured>=0.20.8",          # ドキュメント処理
]
```

### 環境変数設定

`.env`ファイルをプロジェクトルートに作成：

```bash
# HuggingFace認証（モデルダウンロード用）
HF_TOKEN=your_huggingface_token

# Langfuseトレーシング（オプション・現在無効化中）
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# モデルキャッシュディレクトリ（オプション）
HF_HOME=./models  # デフォルト: ~/.cache/huggingface
```

---

## 🚀 使用方法

### 基本的な実行

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
├── your_paper_chunks.json    # 抽出されたチャンク（JSON形式）
└── your_paper_answer.json    # RAG回答（検証結果含む）
```

---

## 🧩 実装詳細

### 主要クラス

#### `AgenticRAGPipeline`

メインのパイプラインクラス。すべてのコンポーネントを統合。

```python
class AgenticRAGPipeline:
    def __init__(
        self,
        pdf_parser: PDFParser,
        router: AgentRouter,
        chunk_store: ChunkStore,
        orchestrator: ReasoningOrchestratorAgent,
        validator: ChunkValidatorAgent | None = None,
        answer_validator: AnswerValidatorAgent | None = None,
        tracer: LangfuseTracer | None = None,
    ):
        ...
    
    def ingest(self, pdf_path: str, validates: bool = False) -> list[ProcessedChunk]:
        """PDFを解析してチャンクを抽出、ベクトルストアに保存"""
        
    def query(self, question: str, validates: bool = False) -> RAGAnswer:
        """RAGクエリを実行し、回答を生成"""
```

#### `ModelCache`

HuggingFace Hub経由でMLXモデルをダウンロード・キャッシュ。

```python
class ModelCache:
    """
    MLXモデルのローカルキャッシュ管理。
    環境変数HF_HOMEで指定されたディレクトリに保存。
    """
    
    @classmethod
    def get_model_path(cls, model_id: str) -> str:
        """
        ローカルキャッシュパスを返すか、ダウンロードを実行。
        
        Returns:
            ローカルモデルディレクトリのパス
        """
```

#### `BaseLoadableModel`

モデルのライフサイクル管理Mixin。

```python
class BaseLoadableModel:
    """
    明示的なload/unloadライフサイクルを持つモデルMixin。
    コンテキストマネージャーとして使用可能。
    """
    
    def load(self) -> None:
        """モデルをメモリにロード"""
        logger.info(f"Loading model: {self.model_id}")
        self.model = mlx_lm.load(self.model_id)
        
    def unload(self) -> None:
        """モデルをメモリから解放"""
        logger.info(f"Unloading model: {self.model_id}")
        self.model = None
        gc.collect()  # ガベージコレクション実行
        
    def __enter__(self):
        self.load()
        return self
        
    def __exit__(self, *args):
        self.unload()
```

#### `ChunkValidatorAgent`

CHECKPOINT A: チャンク品質監査。

```python
class ChunkValidatorAgent(BaseLoadableModel):
    """
    抽出されたチャンクの品質を検証。
    VLMモデルを使用して、図表の場合は画像を直接検証。
    """
    
    def validate(
        self, 
        chunk: ProcessedChunk, 
        raw_chunk: RawChunk
    ) -> ChunkValidationResult:
        """
        チャンクを検証し、必要に応じて修正。
        
        Returns:
            ChunkValidationResult: is_valid, corrected, verdict_score等
        """
```

#### `AnswerValidatorAgent`

CHECKPOINT B: 回答幻覚検出。

```python
class AnswerValidatorAgent(BaseLoadableModel):
    """
    RAG回答の各主張がソースに基づいているか検証。
    根拠のない主張を検出して修正。
    """
    
    def validate(
        self, 
        answer: str, 
        sources: list[ProcessedChunk]
    ) -> AnswerValidationResult:
        """
        回答を検証し、幻覚を検出。
        
        Returns:
            AnswerValidationResult: is_grounded, corrected_answer等
        """
```

---

## 🎨 デザイン判断

### なぜMLX？

1. **Apple Silicon最適化**: M1/M2/M3チップのNeural Engineを最大限活用
2. **メモリ効率**: 4-bit量子化により、VRAM使用量を75%削減（FP16比）
3. **速度**: transformers+PyTorchより2-3倍高速（Apple Silicon上）
4. **シンプルさ**: CUDAやROCmの設定不要

### なぜSequential Loading？

1. **メモリ制約**: 8GBのMacでも動作可能
2. **予測可能性**: VRAMピークが明確（~5GB）
3. **拡張性**: より大型のモデル（14B, 30B）にも対応可能

### なぜ2段階バリデーション？

1. **早期問題検出**: CHECKPOINT Aで不正なチャンクを事前に排除
2. **ユーザー信頼性**: CHECKPOINT Bで最終回答の根拠を保証
3. **デバッグ容易性**: 各段階で品質スコアを記録

---

## 📊 パフォーマンス特性

### ベンチマーク（Apple M2 Max, 32GB RAM）

| 処理 | PDF（20ページ） | メモリピーク | 時間 |
|------|----------------|------------|------|
| **PDF解析** | 12 TEXT + 3 TABLE + 5 FIGURE | ~500MB | ~15秒 |
| **チャンク抽出** | 20チャンク | ~4.5GB | ~45秒 |
| **CHECKPOINT A** | 20チャンク検証 | ~1GB | ~30秒 |
| **ベクトル化** | 20チャンク | ~500MB | ~5秒 |
| **RAG検索** | トップ8検索 | ~500MB | <1秒 |
| **回答生成** | 1クエリ | ~4GB | ~20秒 |
| **CHECKPOINT B** | 1回答検証 | ~4GB | ~15秒 |

**合計処理時間**: 約2分30秒（20ページPDF、フルバリデーション有効）

---

## 🔮 今後の展開

### 予定されている機能

1. **Langfuseトレーシングの再有効化**
   - SDK APIの互換性が復旧次第、再実装

2. **バッチ処理サポート**
   - 複数PDFの一括処理
   - 並列化によるスループット向上

3. **WebUIの追加**
   - Gradioベースのインタラクティブインターフェース
   - リアルタイムプレビューと編集

4. **より大型のモデルオプション**
   - 70B+モデルのサポート（Mac Studio等）
   - API経由での商用モデル統合（Claude, GPT-4）

5. **追加のバリデーター**
   - 事実整合性チェッカー（外部知識ベース）
   - 時系列整合性検証（論文の実験結果等）

---

## 📚 参考資料

### MLX関連
- [MLX公式ドキュメント](https://ml-explore.github.io/mlx/build/html/index.html)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)

### モデル
- [Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

### PDF処理
- [pymupdf (fitz)](https://pymupdf.readthedocs.io/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)

### ベクトルDB
- [ChromaDB](https://www.trychroma.com/)
- [sentence-transformers](https://www.sbert.net/)

---

## 🙏 謝辞

このプロジェクトは、以下のオープンソースプロジェクトの上に成り立っています：

- **MLX** (Apple) - Apple Silicon最適化フレームワーク
- **HuggingFace** - モデルホスティングとトランスフォーマーライブラリ
- **ChromaDB** - 軽量ベクトルストア
- **pymupdf & pdfplumber** - Python PDF処理ライブラリ

---

**Last Updated**: 2026-02-20  
**Version**: 3.0  
**Maintainer**: Agentic RAG Team

