# Configuration Setup Guide

Last updated: 2026-03-15

このガイドは、現行実装に合わせた設定手順をまとめたものです。

## 1. 前提

- Python 3.13+
- macOS (Apple Silicon)
- ローカル実行前提（MLXモデル）

本プロジェクトはローカルMLX実行を基本としており、OpenAI等の外部推論APIキーは必須ではありません。

## 2. 設定ファイル

### 2.1 settings.json

プロジェクトルートの `settings.json` を使用します。

```bash
cp settings.example.json settings.json
```

`src/core/config.py` が起動時に `settings.json` を読み込み、不足キーはデフォルトで補完します。

### 2.2 主なキー

`settings.json` の主なセクション:

- `models`
  - `text_extraction`
  - `table_extraction`
  - `vision_extraction`
  - `orchestrator`
  - `chunk_validator`
  - `answer_validator`
  - `dspy_lm`
  - `embedder`
- `pipeline`
  - `max_context_chunks`
  - `embedder_batch_size`
  - `chunk_size`
- `validation`
  - `confidence_threshold`
  - `enable_checkpoint_a`
  - `enable_checkpoint_b`

## 3. CLIオーバーライド

`settings.json` の値は、CLI引数で一時的に上書きできます。

代表例:

```bash
python app.py ingest input/sample.pdf \
  --text-model mlx-community/Qwen3-8B-4bit \
  --enable-figure-aware-fallback \
  --table-model mlx-community/Qwen2.5-3B-Instruct-4bit
```

query/pipelineでも同様に`--orchestrator-model`などを指定できます。
`--enable-figure-aware-fallback` は ingest / pipeline で利用できます。

## 4. 出力先設定

CLIには `--output` がありますが、用途は2系統です。

- chunks/answer JSON:
  - `src/utils/serialization.py` の `OUTPUT_DIR=./output` に保存
  - `--output` の指定値とは独立
- 監査レポート（audit）:
  - `--output` の指定先を使用

運用上は「JSONは `./output`、監査は `--output`」として扱うと混乱が少なくなります。

## 5. Langfuse設定（任意）

トレースを有効にする場合は環境変数を設定します。

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

未設定でも処理は継続します（no-op動作）。

## 6. モデルキャッシュ

モデルは `src/core/cache.py` によりキャッシュ管理されます。

- 初回ロードは時間がかかる場合があります
- 同一モデルの再利用で起動コストを削減
- 必要に応じて `cleanup_unused_models()` が未使用モデルを解放

## 7. 動作確認

### 7.1 設定の読込確認

```bash
python -c "from src.core.config import config; print(config.get_model('orchestrator'))"
```

### 7.2 CLIヘルプ確認

```bash
python app.py --help
python app.py ingest --help
python app.py query --help
python app.py pipeline --help
```

## 8. よくある注意点

- `settings.json`のJSON構文エラー時はデフォルト値へフォールバックします。
- `--validate` はデフォルトで有効です（`--no-validate` で無効化）。
- `--no-validate` の作用範囲:
  - ingest: チャンク検証（CHECKPOINT A/B）
  - query: 最終回答検証
- `--enable-figure-aware-fallback` の作用範囲:
  - ingest / pipeline: parserのfigure-aware fallback条件を有効化
  - query単体: 未使用
- `--use-langgraph` の適用範囲:
  - ingest: LangGraphIngestPipeline
  - query: LangGraphQueryPipeline
  - pipeline: queryフェーズに適用
- `--use-crewai`はingest/query/pipelineで利用可能です。
