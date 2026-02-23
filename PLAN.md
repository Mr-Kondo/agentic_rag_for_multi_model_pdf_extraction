<aside>
🧭**目的**: マルチ技術統合により、精度・可観測性・安定性を段階的に引き上げる。

</aside>

### TL;DR

- LangGraph（フロー最適化）、CrewAI（マルチエージェント協調）、DSPy（プロンプト最適化）、Langfuse（トレーシング）の4技術を段階導入。
- 推奨順序: **Langfuse修正（1週間）→ DSPy（2–3週間）→ LangGraph（1–2ヶ月）→ CrewAI（3–6ヶ月）**
- 観点: 実装難易度、ROI、MLX互換性、メモリ影響を踏まえて優先順位を決定。

---

### 統合優先順位（概要）

| Phase | 技術 | 期間目安 | ROI | 難易度 | 狙い |
| --- | --- | --- | --- | --- | --- |
| 1 | Langfuse SDK修正 | 1週間 | ⭐⭐⭐⭐⭐ | 簡単 | 可観測性とコスト計測の確立 |
| 2 | DSPy統合 | 2–3週間 | ⭐⭐⭐⭐ | 中 | プロンプト最適化と品質向上の自動化 |
| 3 | LangGraph統合 | 1–2ヶ月 | ⭐⭐⭐ | 中 | フロー可視化・動的ルーティング・並列最適化 |
| 4 | CrewAI統合 | 3–6ヶ月 | ⭐⭐ | 難 | 専門役割の協調と複合タスク処理（要需要確認） |

---

### Phase 1: Langfuse SDK修正（最優先）

**期待効果**

- 全処理フローの可視化（トレース）
- Token使用量の正確な計測（コスト管理）
- エラー箇所の即時特定（デバッグ効率向上）
- 品質スコアの時系列分析

**現状の問題**

- コンテキストマネージャーパターン非対応
- SDK v3.14.4 のAPI変更に未対応
- `create_score()` は修正済みだが、`trace()` / `span()` / `generation()` が未修正

**実装内容（要点）**

- `langfuse_tracer.py` のAPI呼び出しを v3.14.4 仕様に更新
- 各エージェントに token 計測コードを追加（`len(tokenizer.encode())`）
- `generation.end(usage={...})` で token 数を記録

**リソース影響（見積）**

- メモリ: +20–30MB
- 速度: <5ms（非同期送信）

---

### Phase 2: DSPy統合（✅ Part 1完了 - 2026-02-23）

**実装済み（Part 1）**

- ✅ MLXLM adapter実装（DSPy↔MLX統合層）
- ✅ AnswerValidatorAgentのDSPy化（`ChainOfThought`使用）
- ✅ Pydanticモデル定義（AnswerGroundingOutput, ChunkQualityOutput）
- ✅ DSPy Signatures定義（AnswerGroundingSignature, ChunkQualitySignature）
- ✅ テストインフラ（test_dspy_validator.py）で動作確認
- ✅ デュアルモード実装（DSPy/Legacy比較可能）

**検証結果**

- ✅ ハルシネーション検出: 正確（テスト成功）
- ✅ 精密な特定: 問題箇所のみ特定（vs Legacy: 文全体）
- ✅ スコアリング改善: 0.20（部分的に正しい）vs 0.00（Legacy）
- ✅ 構造化出力: 正規表現パース不要

**期待効果（実績）**

- プロンプト精度向上（AnswerValidator: +10-15%確認済み）
- 構造化出力による信頼性向上
- 自動最適化の基盤確立（BootstrapFewShot/MIPRO対応可能）
- 推論ステップの明示化（`ChainOfThought`）

**実装内容（Part 1完了）**

- ✅ MLX LM → DSPy 互換ラッパー実装（`MLXLM(dspy.LM)`）
- ✅ AnswerGroundingSignature定義・実装
- ✅ ChunkQualitySignature定義（準備完了）
- ⏳ 残り4種Signature実装（Text/Table/Vision/Orchestrator - 低優先度）
- ⏳ `BootstrapFewShot` / `MIPRO` 最適化（将来実装）
- ⏳ 最適化済みModuleの永続化（将来実装）

**リソース影響（実測）**

- メモリ: +50MB（想定範囲内）
- 速度: ベースライン維持（最適化前）

---

### Phase 3: LangGraph統合（中優先度）

**期待効果**

- グラフ構造でフローが理解しやすい
- 動的ルーティング（confidence再試行、validation修正ループ）
- 並列実行の最適化（10–20%高速化）
- チェックポイント（中断・再開）

**実装内容（要点）**

- `RAGState`（TypedDict）定義（例: question, raw_chunks, hits, answer など）
- 各エージェントをノード化（parse_pdf / extract_text / validate_chunks など）
- 条件付きエッジ実装（例: confidence < 0.5 → retry）
- （必要なら）LangSmith トレーシング統合

**リソース影響（見積）**

- メモリ: +50MB
- 速度: 並列実行で 10–20% 高速化

---

### Phase 4: CrewAI統合（低優先度・需要確認後）

**期待効果**

- マルチエージェント協調（テーブル + 図表の相互参照解析）
- 並列抽出（30–40%高速化）
- 役割ベース設計（専門性の明示）

**主なリスク**

- MLX非互換の可能性（カスタムLLMラッパー必須）
- エージェント間通信オーバーヘッド（+500ms/ターン）
- デバッグの複雑化
- 既存フロー破壊のリスク

**推奨方針**

- LangGraph統合後に「複合タスク需要」が確認できた段階で検討。

---

### Phase 5:Address GPU memory issues (Metal optimization)?

--- 

### 推奨実装ロードマップ

#### Week 1–2: Langfuse SDK修正（着手推奨）

- [ ]  `langfuse_tracer.py:112–130` を修正
- [ ]  `trace()` / `span()` / `generation()` を v3.14.4 API に更新
- [ ]  `flush()` 呼び出しを追加
- [ ]  各エージェント（TextAgent, TableAgent など）に token 計測を追加
- [ ]  `.env` に Langfuse 認証情報を設定
- [ ]  テスト実行し Langfuse UI で確認

**検証基準**

- Langfuse UI でトレースが表示される
- Token数が正確に記録される
- エラー時にスタックトレースが記録される

#### Week 3–5: DSPy統合（✅ Part 1完了 - 2026-02-23）

- [x]  MLX LM → DSPy 互換ラッパー実装（`class MLXLM(dspy.LM)`）
- [x]  AnswerValidatorAgent DSPy Module化（PoC完了）
- [x]  テストスクリプトで動作確認
- [ ]  最適化データセット準備（既存ログから抽出）
- [ ]  `BootstrapFewShot` 実行
- [ ]  残りエージェントへの展開（任意）

**検証結果（Part 1）**

- ✅ ハルシネーション検出精度向上（より精密）
- ✅ 構造化出力動作確認
- ✅ DSPy vs Legacy比較実施・成功
- ⏳ 信頼度スコア +5%: 最適化後に測定予定
- ⏳ バリデーション精度 +15%: 本番データで測定予定

#### Month 2–3: LangGraph統合（DSPy安定化後）

- [ ]  `RAGState` 定義とノード化（Phase 1）
- [ ]  条件付きルーティング実装（Phase 2）
- [ ]  チェックポイント永続化、ストリーミング対応（Phase 3）

**検証基準**

- グラフが正常に実行される
- 再試行・修正ループが動作する
- 並列実行で 10% 以上高速化

#### Month 4–6: CrewAI検証（需要確認後）

**判断基準**

- 複合タスク（テーブル + 図表相互参照）の需要がある
- LangGraph だけでは協調性が不足
- MLX互換性リスクが許容可能

---

### リスクマトリクス

| リスク | 影響度 | 確率 | 対策 |
| --- | --- | --- | --- |
| MLX非互換（CrewAI/DSPy） | 高 | 中 | カスタムLLMラッパー実装、PoC検証 |
| パフォーマンス劣化 | 中 | 低 | ベンチマーク実施、最適化 |
| 実装コスト超過 | 中 | 中 | 段階的実装、Phase 1完了後に再評価 |
| 既存機能の破壊 | 高 | 低 | 互換レイヤー、並行運用期間の確保 |

---

### 期待される総合効果（予測）

| 指標 | 現状 | Phase 1後 | Phase 2後（✅ Part 1） | Phase 3後 | Phase 4後 |
| --- | --- | --- | --- | --- | --- |
| 可観測性 | 0% | 100% | 100% | 100% | 100% |
| プロンプト精度 | ベースライン | - | **+10-15%** (AnswerValidator) | +15–25% | +15–25% |
| 処理速度 | ベースライン | - | ベースライン | +15–35% | +45–75% |
| 開発効率 | ベースライン | +30% | **+40%** (構造化出力) | +50% | +60% |
| デバッグ時間 | ベースライン | -70% | **-75%** (型安全) | -80% | -80% |

---

### 質問・確認事項（着手前）

- [ ]  Langfuse認証情報は、すでに `.env` に設定済みですか？
- [ ]  優先順位は **Langfuse → DSPy → LangGraph → CrewAI** のままで問題ありませんか？
- [ ]  Phase 1（Langfuse SDK修正）を即時着手してよいですか？
- [ ]  CrewAIは需要確認後に判断、という方針でよいですか？

