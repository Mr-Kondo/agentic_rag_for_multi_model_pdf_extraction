# Processing Flow (Mermaid)

Last updated: 2026-03-29

このドキュメントは、CLI 実行時の主要な分岐と処理フローを可視化したものです。実装の責務や制約は [docs/ARCHITECTURE.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/ARCHITECTURE.md)、設定の注意点は [docs/CONFIG_SETUP.md](/Volumes/SSD/Programming/agentic_rag_for_multi_model_pdf_extraction/docs/CONFIG_SETUP.md) を参照してください。

## 1. CLIディスパッチ全体

```mermaid
flowchart TD
  A[CLI start app.py] --> B{command}
  B -->|ingest| C[cmd_ingest]
  B -->|query| D[cmd_query]
  B -->|pipeline| E[cmd_pipeline]

  C --> C1{use_langgraph?}
  C1 -->|yes| C2[LangGraphIngestPipeline.ingest]
  C1 -->|no| C3{use_crewai?}
  C3 -->|yes| C4[AgenticRAGPipeline.ingest_with_crewai]
  C3 -->|no| C5[AgenticRAGPipeline.ingest]

  D --> D1{use_langgraph?}
  D1 -->|yes| D2[LangGraphQueryPipeline.query]
  D1 -->|no| D3{use_crewai?}
  D3 -->|yes| D4[AgenticRAGPipeline.query_with_crewai]
  D3 -->|no| D5[AgenticRAGPipeline.query]

  E --> E1[ingest phase]
  E1 --> E2{use_crewai?}
  E2 -->|yes| E3[ingest_with_crewai]
  E2 -->|no| E4[ingest]
  E3 --> E5[query phase]
  E4 --> E5
  E5 --> E6{use_langgraph?}
  E6 -->|yes| E7[LangGraphQueryPipeline.query]
  E6 -->|no| E8{use_crewai?}
  E8 -->|yes| E9[query_with_crewai]
  E8 -->|no| E10[query]
```

実装参照:
- app.py: cmd_ingest, cmd_query, cmd_pipeline

注記:

- `pipeline` で `--use-crewai` と `--use-langgraph` を同時指定した場合、query フェーズも CrewAI が優先されます。
- `pipeline --use-langgraph` は ingest を LangGraph 化しません。

## 2. ingest 詳細フロー

```mermaid
flowchart TD
  A[ingest start] --> B{use_langgraph?}

  B -->|yes| LG0[Build LangGraphIngestPipeline]
  LG0 --> LG1[PDFParser.parse]
  LG1 --> LG2[extract via graph nodes]
  LG2 --> LG3{validate?}
  LG3 -->|yes| LG4[Checkpoint A chunk validation]
  LG3 -->|no| LG5[Skip validation]
  LG4 --> LG6[store chunks to Chroma]
  LG5 --> LG6
  LG6 --> LG7[save chunks JSON]

  B -->|no| C{use_crewai?}
  C -->|yes| CR0[Build AgenticRAGPipeline use_crewai]
  CR0 --> CR1[PDFParser.parse]
  CR1 --> CR2[CrewAI ingest wrapper then local AgentRouter extraction]
  CR2 --> CR3[Cross link relationships]
  CR3 --> CR4{validate?}
  CR4 -->|yes| CR5[Checkpoint A chunk validation]
  CR4 -->|no| CR6[Skip validation]
  CR5 --> CR7[store chunks to Chroma]
  CR6 --> CR7
  CR7 --> CR8[save chunks JSON]

  C -->|no| SQ0[Build AgenticRAGPipeline sequential]
  SQ0 --> SQ1[PDFParser.parse]
  SQ1 --> SQ2[AgentRouter.extract_chunks]
  SQ2 --> SQ3{validate?}
  SQ3 -->|yes| SQ4[Checkpoint A chunk validation]
  SQ3 -->|no| SQ5[Skip validation]
  SQ4 --> SQ6[store chunks to Chroma]
  SQ5 --> SQ6
  SQ6 --> SQ7[save chunks JSON]

  F1[enable_figure_aware_fallback] --> SQ1
  F1 --> CR1
  F1 --> LG1
```

実装参照:
- app.py: cmd_ingest
- src/core/pipeline.py: build, ingest, ingest_with_crewai
- src/core/crewai_pipeline.py: CrewAIIngestionPipeline.process_chunks
- src/core/langgraph_pipeline.py: LangGraphIngestPipeline.build, ingest
- src/core/parser.py: PDFParser

注記:

- CrewAI ingest の extraction は、現状では Crew task による全面抽出ではなく、`ExtractionCrew.extract_chunks()` 内で local `AgentRouter` を直接呼びます。
- CrewAI ingest の validation / linking は簡略実装を含みます。

## 3. query 詳細フロー

```mermaid
flowchart TD
  A[query start] --> B{use_langgraph?}

  B -->|yes| LG0[Build LangGraphQueryPipeline]
  LG0 --> LG1[retrieve relevant chunks]
  LG1 --> LG2[generate answer]
  LG2 --> LG3{validate?}
  LG3 -->|yes| LG4[Checkpoint B answer validation and optional revise]
  LG3 -->|no| LG5[Skip validation]
  LG4 --> LG6[return answer and metadata]
  LG5 --> LG6

  B -->|no| C{use_crewai?}
  C -->|yes| CR0[Build AgenticRAGPipeline use_crewai]
  CR0 --> CR1[retrieve from Chroma]
  CR1 --> CR2[CrewAI query flow]
  CR2 --> CR3{validate?}
  CR3 -->|yes| CR4[Checkpoint B answer validation]
  CR3 -->|no| CR5[Skip validation]
  CR4 --> CR6[return answer and metadata]
  CR5 --> CR6

  C -->|no| SQ0[Build AgenticRAGPipeline sequential]
  SQ0 --> SQ1[retrieve from Chroma]
  SQ1 --> SQ2[orchestrator generate answer]
  SQ2 --> SQ3{validate?}
  SQ3 -->|yes| SQ4[Checkpoint B answer validation]
  SQ3 -->|no| SQ5[Skip validation]
  SQ4 --> SQ6[return answer and metadata]
  SQ5 --> SQ6
```

実装参照:
- app.py: cmd_query
- src/core/pipeline.py: query, query_with_crewai
- src/core/langgraph_pipeline.py: LangGraphQueryPipeline.query

注記:

- CrewAI query は失敗時に標準 query へフォールバックします。

## 4. pipeline 詳細フロー

```mermaid
flowchart TD
  A[pipeline start] --> B[build AgenticRAGPipeline]
  B --> C{use_crewai for ingest?}
  C -->|yes| D[ingest_with_crewai]
  C -->|no| E[ingest sequential]

  D --> F{query mode}
  E --> F

  F -->|use_langgraph| G[build LangGraphQueryPipeline and query]
  F -->|use_crewai| H[query_with_crewai]
  F -->|default| I[query sequential]

  G --> J[save answer JSON]
  H --> J
  I --> J
```

実装参照:
- app.py: cmd_pipeline
- src/core/pipeline.py: ingest, ingest_with_crewai, query, query_with_crewai
- src/core/langgraph_pipeline.py: LangGraphQueryPipeline.build, query

注記:

- `pipeline` は ingest 側に LangGraph 経路を持たず、ingest は Sequential か CrewAI のどちらかです。

## 5. LangGraph query ノード遷移

```mermaid
flowchart TD
  A[START] --> B[retrieve]
  B --> C[check_quality]
  C --> D[generate]
  D --> E[decide_validate]
  E -->|skip| I[finalize]
  E -->|validate| F[validate]
  F --> G[check_grounding]
  G -->|grounded| I
  G -->|not grounded and attempts left| H[revise]
  H --> F
  G -->|max attempts| I
  I --> J[END]
```

実装参照:
- src/core/langgraph_pipeline.py: _build_graph, route_after_quality_check, route_after_decide_validate, route_after_grounding_check

## 更新ルール

- CLIオプションの追加や意味変更があった場合は、この文書の該当図を更新する。
- 変更対象の目安:
  - ingest分岐変更: セクション2
  - query分岐変更: セクション3
  - pipeline分岐変更: セクション4
  - LangGraphノード遷移変更: セクション5
- READMEには概要のみ残し、処理追跡の詳細は本ドキュメントを正とする。
