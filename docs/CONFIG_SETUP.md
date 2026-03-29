# Configuration Setup Guide

Last updated: 2026-03-29

このガイドは、現行実装で有効な設定と、設定ファイルでは見えていてもまだ実運用に接続されていない項目を区別して整理したものです。

## 1. 前提

- Python 3.13以上
- macOS / Apple Silicon
- ローカルMLX実行前提

OpenAIなどの外部推論APIは必須ではありません。

## 2. `settings.json`

```bash
cp settings.example.json settings.json
```

`src/core/config.py` の `ConfigLoader` は次の順で設定を作ります。

1. コード内 `_DEFAULTS`
2. `settings.json`の内容をdeep merge
3. 呼び出し側が明示的に渡した引数

### 2.1 deep merge の意味

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

### 2.2 主なキー

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
- `cache`
  - `enable_hf_cache`
  - `cache_dir`
- `ocr`
  - `engine`
  - `default_lang`
  - `japanese_lang`
  - `fallback_lang`
  - `config`
- `validation`
  - `confidence_threshold`
  - `enable_checkpoint_a`
  - `enable_checkpoint_b`

## 3. 現在の優先順位と制約

### 3.1 CLI から見たモデル設定

ここが一番誤解しやすい点です。

- `src/core/config.py`にはmodelの既定値があります。
- しかし`app.py`のCLI引数にもmodelの既定値がハードコードされています。
- `app.py`はその既定値を`AgenticRAGPipeline.build()`へ渡すため、CLIで実行する限り、未指定時でも`settings.json`の`models.*`がそのまま採用されるわけではありません。

つまり、現行実装のCLIでは次の理解が正確です。

1. 明示的なCLI引数
2. `app.py` の `DEFAULT_MODELS`
3. `settings.json`のmodel値は下位レベルのbuildを直接使う場合に効く

この挙動はOCRやvalidation設定とは異なります。

### 3.2 OCR / validation / parser 設定

これらは `src/core.config.config` を通じて参照されるため、`settings.json` の値が反映されます。

主な対象:

- `ocr.*`
- `validation.*`
- parser内で参照する一部設定

## 4. OCR 設定

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
- `default_lang`などはparserのOCR系定数に取り込まれます。
- Tesseractを補助的に使う場合は`jpn`言語データの導入が必要です。

## 5. `--output` と保存先

CLIの`--output`は名前ほど単純ではありません。

- chunks / answer JSON
  - `src/utils/serialization.py` の `OUTPUT_DIR = ./output` に固定保存されます。
  - `--output` で保存先は変わりません。
- 監査レポート
  - `save_chunk_audit()` に渡され、`--output` の値が反映されます。

したがって、現状の運用ルールは次の理解が安全です。

- JSONは`./output`
- auditは`--output`

## 6. モデルキャッシュ

`src/core/cache.py`はimport時に次を行います。

- `MODEL_CACHE_DIR = ~/.models`
- `os.environ["HF_HOME"] = ~/.models`

このため、`settings.json` の `cache.cache_dir` は現行コードでは実際の保存先切り替えに使われていません。`cache` セクションは将来拡張の余地はありますが、現在の実行上の正本は `src/core/cache.py` です。

`cleanup_unused_models()` は未使用モデルディレクトリの削除を試みます。

## 7. `--lazy-agents`

CLIには`--lazy-agents`がありますが、現時点では次の状態です。

- `AgenticRAGPipeline.build()` に値は渡される
- pipelineオブジェクトに`lazy_agents`として保持される
- ただし、実行時に明確なload / unload分岐へ接続されている箇所は確認できません

そのため、現段階では「将来のメモリ最適化用フラグが公開されている」と理解するのが妥当です。

## 8. Langfuse

環境変数を設定するとtraceが記録されます。

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

未設定時もno-opで動作継続します。

`--session-id`はquery / pipeline実行時に関連トレースを束ねるために使います。

## 9. 動作確認

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

## 10. 注意点

- `settings.json`のJSONが壊れている場合、`ConfigLoader`はログを出してdefaultsにフォールバックします。
- `--validate` は既定で有効です。
- `--enable-figure-aware-fallback`はingest / pipelineのparserにしか効きません。
- `--use-langgraph`はpipelineではqueryフェーズにしか効きません。
- `--use-crewai`はingest / query / pipelineで使えますが、ingestの抽出本体はlocal routerベースです。
