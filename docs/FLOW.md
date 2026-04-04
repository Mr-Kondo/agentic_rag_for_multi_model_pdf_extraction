# Processing Flow

Last updated: 2026-04-04

このドキュメントは、現在サポートされている Sequential CLI flow だけを可視化します。詳細な責務は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、設定上の注意点は [docs/CONFIG_SETUP.md](docs/CONFIG_SETUP.md) を参照してください。

## 1. CLI dispatch

```mermaid
flowchart TD
  A[CLI start app.py] --> B{command}
  B -->|ingest| C[cmd_ingest]
  B -->|query| D[cmd_query]
  B -->|pipeline| E[cmd_pipeline]

  C --> F[AgenticRAGPipeline.build]
  F --> G[AgenticRAGPipeline.ingest]

  D --> H[AgenticRAGPipeline.build]
  H --> I[AgenticRAGPipeline.query]

  E --> J[AgenticRAGPipeline.build]
  J --> K[AgenticRAGPipeline.ingest]
  K --> L[AgenticRAGPipeline.query]
```

実装参照:
- `app.py`: `cmd_ingest`, `cmd_query`, `cmd_pipeline`

## 2. Ingest flow

```mermaid
flowchart TD
  A[ingest start] --> B[PDFParser.parse]
  B --> C[AgentRouter.route_with_policy per chunk]
  C --> D{validate?}
  D -->|yes| E[ChunkValidatorAgent validate_chunk]
  D -->|no| F[confidence threshold filter]
  E --> G[accepted chunks]
  F --> G
  G --> H[ChunkStore.upsert]
  H --> I{audit output requested?}
  I -->|yes| J[save_chunk_audit]
  I -->|no| K[return chunks]
  J --> K
```

実装参照:
- `src/core/pipeline.py`: `ingest`
- `src/core/parser.py`: `PDFParser`
- `src/agents/router.py`: `AgentRouter`

## 3. Query flow

```mermaid
flowchart TD
  A[query start] --> B[retrieve from ChromaDB]
  B --> C[ReasoningOrchestratorAgent.generate]
  C --> D{validate?}
  D -->|yes| E[AnswerValidatorAgent.validate_answer]
  D -->|no| F[return generated answer]
  E --> G{grounded?}
  G -->|yes| H[return validated answer]
  G -->|no with revised answer| I[substitute revised answer]
  G -->|no without revised answer| J[return original answer with validation summary]
  I --> H
```

実装参照:
- `src/core/pipeline.py`: `query`
- `src/agents/orchestrator.py`: `ReasoningOrchestratorAgent`
- `src/agents/validation.py`: `AnswerValidatorAgent`

## 4. Combined pipeline flow

```mermaid
flowchart TD
  A[pipeline start] --> B[build pipeline]
  B --> C[ingest PDF]
  C --> D[query stored chunks]
  D --> E[save answer and audit outputs]
```

実装参照:
- `app.py`: `cmd_pipeline`
- `src/core/pipeline.py`: `ingest`, `query`

## 5. Notes

- サポートされる実行モードは Sequential のみです。
- `--enable-figure-aware-fallback` は ingest 側 parser にだけ影響します。
- `--validate` は ingest では CHECKPOINT A、query では CHECKPOINT B を制御します。
- README には概要だけを残し、処理の詳細はこの文書を正とします。
