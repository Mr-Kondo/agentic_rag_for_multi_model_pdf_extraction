# Configuration Setup Guide

Last updated: 2026-04-19

このガイドは、現行の Sequential + Ollama runtime で有効な設定のみを整理します。

## 1. Prerequisites

- Python 3.13+
- Ollama server
- ChromaDB 保存先への書き込み権限
- OCR 利用時のエンジン依存
  - easyocr: Python package
  - yomitoku: Python package + 初回モデル取得
  - tesseract: バイナリ + 言語データ

## 2. settings.json の準備

```bash
cp settings.example.json settings.json
```

`src/core/config.py` の `ConfigLoader` は次の順で設定を解決します。

1. `_DEFAULTS`
2. `settings.json` を deep merge
3. 実行時 CLI 引数

## 3. 設定優先順位の詳細

### 3.1 モデル設定

CLI のモデル引数 default は `config.get_model(...)` で作られるため、
`settings.json` が存在すると help 表示の default もその値になります。

実質優先順位:

1. 明示 CLI (`--text-model` 等)
2. `settings.json` の `models.*`
3. `_DEFAULTS` の `models.*`

### 3.2 validate 制御

現行実装では、checkpoint の有効/無効は `--validate` / `--no-validate` が主制御です。

- ingest: CHECKPOINT A
- query: CHECKPOINT B
- pipeline: 両方

`validation.enable_checkpoint_a` / `validation.enable_checkpoint_b` は設定キーとして存在しますが、
現状の runtime 分岐は CLI の `validate` フラグ中心です。

## 4. Main keys

- `ollama`
  - `base_url`
  - `request_timeout_seconds`
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
  - `backend` (`ollama` | `sentence_transformers`)
  - `query_prefix`
  - `passage_prefix`
  - `batch_size`
  - `max_input_chars`
  - `retry_trim_enabled`
  - `retry_trim_min_chars`
- `pipeline`
  - `max_context_chunks`
  - `embedder_batch_size`
  - `chunk_size`
  - `text_passthrough`
  - `figure_ocr_only`
- `parser`
  - `enable_native_pdf_heuristic`
  - `native_pdf_max_images`
  - `native_pdf_min_words`
- `ocr`
  - `engine` (`easyocr` | `yomitoku` | `tesseract`)
  - `yomitoku_device`
  - `prewarm_yomitoku`
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
  - `region_policies`
  - `post_correction`
- `audit`
  - `render_page_previews`
- `validation`
  - `confidence_threshold`
  - `enable_checkpoint_a`
  - `enable_checkpoint_b`
- `cache`
  - `enable_hf_cache`
  - `cache_dir`

## 5. OCR 設定

`ocr.engine` は global default、`ocr.region_policies` は chunk type ごとの上書きです。

- text/table/figure で engine を分離可能
- `yomitoku` と `easyocr` は空結果時に tesseract fallback が走る
- figure default policy は `tesseract`

Tesseract 日本語例:

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs
```

詳細は [docs/OCR_ENGINE_BEHAVIOR.md](docs/OCR_ENGINE_BEHAVIOR.md) を参照。

## 6. --output の扱い

CLI の `--output` default は `./out` です。
この設定は chunks / answer / audit の保存先に共通で使われます。

## 7. cache 設定の注意

`cache.cache_dir` キーは存在しますが、
現状の sentence-transformers cache は `src/core/cache.py` で `HF_HOME=~/.models` を設定します。

## 8. settings.example.json と _DEFAULTS の差

`settings.example.json` は推奨構成のサンプルで、`_DEFAULTS` と異なる値を含みます。

例:

- `_DEFAULTS.models.text_extraction`: `qwen3:30b`
- `settings.example.json models.text_extraction`: `qwen3.5:latest`

つまり `settings.json` が存在しない場合は `_DEFAULTS` が使われ、
`settings.json` を作成するとサンプル値へ寄せる運用ができます。

## 9. Smoke checks

```bash
uv run python app.py --help
uv run python app.py ingest --help
uv run python app.py query --help
uv run python app.py pipeline --help
```

```bash
uv run python -c "from src.core.config import config; print(config.get('ocr.engine'))"
```

## 10. Related docs

- [docs/CLI_BEHAVIOR_REFERENCE.md](docs/CLI_BEHAVIOR_REFERENCE.md)
- [docs/OCR_ENGINE_BEHAVIOR.md](docs/OCR_ENGINE_BEHAVIOR.md)
- [docs/KNOWN_CAVEATS.md](docs/KNOWN_CAVEATS.md)
