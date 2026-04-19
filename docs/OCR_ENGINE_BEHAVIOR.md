# OCR Engine Behavior

Last updated: 2026-04-19

この文書は parser の OCR エンジン選択とフォールバック挙動を実装準拠で説明します。

## 1. Supported engines

- `easyocr`
- `yomitoku`
- `tesseract`

設定キー:

- `ocr.engine`
- `ocr.region_policies.text.engine`
- `ocr.region_policies.table.engine`
- `ocr.region_policies.figure.engine`

## 2. Dispatch behavior

`PDFParser._ocr_text_from_bbox()` は次の順で分岐します。

1. `active_policy.engine == "yomitoku"`
   - `_ocr_text_from_bbox_yomitoku()` を実行
   - 空結果なら tesseract fallback
2. `active_policy.engine == "easyocr"`
   - `_ocr_text_from_bbox_easyocr()` を実行
   - 空結果なら tesseract fallback
3. それ以外
   - `_ocr_text_from_bbox_tesseract()` を実行

## 3. Engine-specific settings

### 3.1 easyocr

- `ocr.prewarm_easyocr`
- `ocr.line_confidence_threshold`
- `ocr.enable_line_reocr`
- `ocr.max_line_reocr_attempts`

### 3.2 yomitoku

- `ocr.yomitoku_device` (default: `mps`)
- `ocr.prewarm_yomitoku`
- yomitoku 返却値は line grouping のため EasyOCR 互換 tuple に変換される

### 3.3 tesseract

- `ocr.default_lang`
- `ocr.japanese_lang`
- `ocr.fallback_lang`
- `ocr.config`
- `ocr.tesseract_render_scale`

## 4. Region policy

`ocr.region_policies` で chunk type ごとに設定します。

- `text`
- `table`
- `figure`

各 policy で設定可能:

- `engine`
- `line_confidence_threshold`
- `enable_reocr`
- `max_reocr_attempts`
- `apply_post_correction`

## 5. Operational prerequisites

### 5.1 easyocr

- Python package と依存ランタイム

### 5.2 yomitoku

- Python package
- 初回実行時のモデル取得

### 5.3 tesseract

```bash
brew install tesseract
brew install tesseract-lang
tesseract --list-langs
```

## 6. Common failure modes

- エンジン import 失敗
  - parser は warning を出して fallback 経路へ移行
- 画像レンダリング失敗
  - その領域は OCR 結果なしとして扱う
- 言語データ不足
  - tesseract の認識率が低下または失敗

## 7. Tuning guidance

- 日本語抽出優先
  - `text` / `table` を `yomitoku` に設定
- 安定運用優先
  - `easyocr` + `tesseract` fallback
- 高速寄り
  - `line_reocr` の回数を減らす

## 8. Related docs

- [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md)
- [docs/KNOWN_CAVEATS.md](docs/KNOWN_CAVEATS.md)
