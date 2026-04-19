# CLI Behavior Reference

Last updated: 2026-04-19

この文書は CLI 実装の正確な挙動をまとめたリファレンスです。

## 1. Commands

- `ingest <pdf_path>`
- `query "<question>"`
- `pipeline <pdf_path> "<question>"`

## 2. Shared options

- `--storage-dir` (default: `./chroma_db`)
- `--output` (default: `./out`)
- `--lazy-agents`
- `--quality-mode` / `--fast-mode` (mutually exclusive)
- model options (`--text-model`, `--table-model`, `--vision-model`, `--orchestrator-model`, `--chunk-validator-model`, `--answer-validator-model`)

## 3. Validation flags

- ingest
  - `--validate` / `--no-validate` -> CHECKPOINT A 制御
- query
  - `--validate` / `--no-validate` -> CHECKPOINT B 制御
- pipeline
  - `--validate` / `--no-validate` -> A/B 両方制御

## 4. Runtime mode flags

- `--quality-mode`
  - 実行時に `pipeline.text_passthrough=False` を適用
- `--fast-mode`
  - 実行時に `pipeline.text_passthrough=True` を適用

## 5. Output artifacts by command

`--output` は chunks/answer/audit の共通保存先です。

### 5.1 ingest

- `<pdf_stem>_chunks.json`
- `<pdf_stem>_audit.json`
- `<pdf_stem>_audit.html`
- `<pdf_stem>_audit/pages/*.png`
- `<pdf_stem>_audit/figures/*.png`

### 5.2 query

- `query_answer.json`

注: query 実装は保存時に `Path("query.pdf")` を使うため、answer 名は固定で `query_answer.json` です。

### 5.3 pipeline

- `<pdf_stem>_chunks.json`
- `<pdf_stem>_answer.json`
- `<pdf_stem>_audit.json`
- `<pdf_stem>_audit.html`
- `<pdf_stem>_audit/pages/*.png`
- `<pdf_stem>_audit/figures/*.png`

## 6. Option applicability

- `--session-id`: query / pipeline
- `--enable-figure-aware-fallback`: ingest / pipeline

## 7. Caveat

`--lazy-agents` は pipeline に渡されますが、現時点では実行分岐への影響が限定的です。
詳細は [docs/KNOWN_CAVEATS.md](docs/KNOWN_CAVEATS.md) を参照してください。

## 8. Verification commands

```bash
uv run python app.py --help
uv run python app.py ingest --help
uv run python app.py query --help
uv run python app.py pipeline --help
```
