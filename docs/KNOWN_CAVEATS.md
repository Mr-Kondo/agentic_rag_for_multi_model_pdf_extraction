# Known Caveats

Last updated: 2026-04-19

この文書は現行実装で誤解されやすい挙動を整理します。

## 1. settings.example.json と _DEFAULTS の差

`settings.example.json` は推奨サンプルであり、`src/core/config.py` の `_DEFAULTS` と値が一致しない項目があります。

例:

- `_DEFAULTS.models.text_extraction`: `qwen3:30b`
- `settings.example.json models.text_extraction`: `qwen3.5:latest`

`settings.json` を作成しない場合は `_DEFAULTS` が使われます。

## 2. validate 制御の実態

`validation.enable_checkpoint_a` / `validation.enable_checkpoint_b` は設定キーとして存在しますが、
現行の実行制御は CLI の `--validate` / `--no-validate` が中心です。

## 3. lazy-agents の現状

`--lazy-agents` フラグは pipeline に渡され保持されますが、
現時点では大きな挙動分岐への接続が限定的です。

## 4. query 出力ファイル名

query コマンドは保存時に `query.pdf` を stem として使うため、
answer 出力名は `query_answer.json` になります。

## 5. cache.cache_dir の適用範囲

`cache.cache_dir` キーは存在しますが、
sentence-transformers cache は現状 `HF_HOME=~/.models` を使います。

## 6. OCR fallback の理解

`yomitoku` / `easyocr` は空結果時に `tesseract` へフォールバックします。
このため、global engine と実際に最終採用される OCR 結果が一致しないケースがあります。

## 7. 推奨運用

- 仕様判断は CLI help と実行結果ファイルで確認する
- 変更時は docs と合わせて最小の smoke test を回す
- OCR 調整時は region policy を chunk type 単位で変更する

## 8. Related docs

- [docs/CLI_BEHAVIOR_REFERENCE.md](docs/CLI_BEHAVIOR_REFERENCE.md)
- [docs/OCR_ENGINE_BEHAVIOR.md](docs/OCR_ENGINE_BEHAVIOR.md)
- [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md)
