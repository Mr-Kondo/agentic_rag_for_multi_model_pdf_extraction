# Configuration Setup Guide

Last updated: 2026-04-16

このガイドは、現行の Sequential + Ollama runtime で有効な設定だけを整理したものです。

## 1. Prerequisites

- Python 3.13 以上
- Ollama server
- ChromaDB 保存先の書き込み権限
- OCR を使う場合は EasyOCR または Tesseract の実行環境

OpenAI などの外部推論 API は必須ではありません。

## 2. `settings.json`

```bash
cp settings.example.json settings.json
```

`src/core/config.py` の `ConfigLoader` は次の順で設定を作ります。

1. コード内 `_DEFAULTS`
2. `settings.json` の内容を deep merge
3. 呼び出し側が明示的に渡した引数

### 2.1 Deep merge

`settings.json` に一部のキーしか書かれていなくても、不足分は `_DEFAULTS` から補完されます。

例:

```json
{
  "ocr": {
    "engine": "easyocr"
  }
}
```

この場合でも `ocr.default_lang` や `ocr.fallback_lang` は既定値のまま残ります。

### 2.2 Main keys

- `ollama`
  - `base_url`
  - `request_timeout_seconds` (Ollama API 呼び出しのタイムアウト秒数、既定: `120`)
- `models`
  - `text_extraction`
  - `table_extraction`
  - `vision_extraction`
  - `orchestrator`
  - `chunk_validator`
  - `answer_validator`
  - `dspy_lm`
  - `embedder`
- `embedder`
  - `backend` (`"sentence_transformers"` または `"ollama"`)
  - `query_prefix` (null = backend 既定値を使用)
  - `passage_prefix` (null = backend 既定値を使用)
  - `batch_size`
  - `max_input_chars` (埋め込み入力の最大文字数、超過時に trim してリトライ; 既定: `800`)
  - `retry_trim_enabled` (trim リトライの有効化; 既定: `true`)
  - `retry_trim_min_chars` (trim リトライの最低文字数; 既定: `128`)
- `pipeline`
  - `max_context_chunks`
  - `embedder_batch_size`
  - `chunk_size`
  - `text_passthrough` (テキストチャンクを LLM に通さず原文のまま採用; 既定: `true`)
  - `figure_ocr_only` (図版チャンクを LLM 解析せず OCR テキストのみ採用; 既定: `true`)
- `cache`
  - `enable_hf_cache`
  - `cache_dir`
- `parser`
  - `enable_native_pdf_heuristic`
  - `native_pdf_max_images`
  - `native_pdf_min_words`
- `ocr`
  - `engine`
  - `default_lang`
  - `japanese_lang`
  - `fallback_lang`
  - `config`
  - `render_scale`
  - `tesseract_render_scale`
  - `prewarm_easyocr`
  - `max_rescue_blocks_per_page`
  - `min_reocr_text_length`
  - `cache_tesseract_languages`
  - `line_confidence_threshold`
  - `enable_line_reocr`
  - `max_line_reocr_attempts`
  - `region_policies` (チャンク種別ごとの OCR ポリシー: `text` / `table` / `figure`)
  - `post_correction` (`enabled`, `dictionary_paths`, `apply_to_ocr_only`)
- `audit`
  - `render_page_previews`
- `validation`
  - `confidence_threshold`
  - `enable_checkpoint_a`
  - `enable_checkpoint_b`

## 3. Configuration precedence

### 3.1 Model selection from the CLI

注意点は、CLI 実行時のモデル値です。

- `src/core/config.py` に defaults があります。
- `app.py` の CLI 引数の既定値は `config.get_model(...)` で解決されます。
- そのため CLI でモデル引数を省略した場合は `settings.json` の `models.*` が使われます。

つまり優先順位は次です。

1. 明示的な CLI 引数
2. `settings.json` の model 値 (CLI 既定値の解決元)
3. `_DEFAULTS` (settings.json が無い/不足している場合)

一方で OCR や validation の設定は `settings.json` から反映されます。

### 3.2 OCR / validation / parser settings

主な対象:

- `ocr.*`
- `validation.*`
- parser が参照する設定項目

## 4. OCR settings

`settings.example.json` の既定値:

```json
"ocr": {
  "engine": "easyocr",
  "default_lang": "jpn+eng",
  "japanese_lang": "jpn",
  "fallback_lang": "eng",
  "config": "--oem 3 --psm 6"
}
```

運用上の注意:

- `engine` の既定値は `easyocr` です。
- Tesseract を補助的に使う場合は `jpn` 言語データの導入が必要です。
- `--enable-figure-aware-fallback` は ingest / pipeline の parser にだけ影響します。

## 5. `--output` and save locations

CLI の `--output` は chunks / answer JSON と audit の両方の保存先です。

- 既定値は `./out`
- `src/utils/serialization.py` の JSON 出力に反映されます
- `save_chunk_audit()` の監査出力にも反映されます

安全な理解は次です。

- `--output` 未指定: `./out`
- `--output` 指定: 指定ディレクトリ

## 6. Model cache and local files

`src/core/cache.py` は import 時に次を行います。

- `MODEL_CACHE_DIR = ~/.models`
- `os.environ["HF_HOME"] = ~/.models`

これは sentence-transformers の embedder cache 用です。LLM 自体のロードとメモリ管理は Ollama server が行います。

`settings.json` の `cache.cache_dir` は現時点では実際の保存先切り替えに使われていません。

## 7. `--lazy-agents`

CLI には `--lazy-agents` がありますが、現時点では大きな実行分岐に接続されていません。

- build に値は渡される
- pipeline object に保持される
- 明確な extra load / unload policy にはまだ接続されていない

## 8. Langfuse

環境変数を設定すると trace が記録されます。

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

未設定時も no-op で処理は継続します。

`--session-id` は query / pipeline 実行時に関連 trace を束ねるために使います。

## 9. Smoke checks

```bash
python app.py --help
python app.py ingest --help
python app.py query --help
python app.py pipeline --help
```

設定読み込みの簡易確認例:

```bash
python -c "from src.core.config import config; print(config.get('ocr.engine'))"
```

## 10. Migration note

以下は削除済みです。

- `--use-crewai`
- `--use-langgraph`
- MLX backend 前提の model IDs や setup 手順
- CrewAI / LangGraph 専用 pipeline 設定

現時点でサポートされる runtime は Sequential + Ollama のみです。

## 11. VLM の選択

`models.vision_extraction` と `models.chunk_validator` はどちらも VLM を使います。
現在の推奨値は `settings.example.json` の `qwen3vl:latest` です。

利用可能な VLM であれば自由に切り替えられます。例えば Ricoh VLM (`ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227`) を使う場合:

```json
{
  "models": {
    "vision_extraction": "ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227",
    "chunk_validator": "ricoh-ai/Qwen-3-VL-Ricoh-8B-20260227"
  }
}
```

注意: Ricoh VLM は利用規約への同意が必要な場合があります。`ollama pull` が失敗する場合は `qwen3vl:latest` に戻してください。

ロールバック:

```json
{
  "models": {
    "vision_extraction": "qwen3vl:latest",
    "chunk_validator": "qwen3vl:latest"
  }
}
```
