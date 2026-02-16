# Design Documentation

## C4 Model Diagrams

All diagrams use [Mermaid.js](https://mermaid.js.org/) and render natively on GitHub.

| Level | File | What it shows |
|-------|------|---------------|
| 1 — Context | [01-context.md](c4-diagrams/01-context.md) | System boundary, external actors, data flow latency targets |
| 2 — Container | [02-container.md](c4-diagrams/02-container.md) | Deployable units: ingestion, storage, retrieval, vector store |
| 3 — Component | [03-component.md](c4-diagrams/03-component.md) | Internal classes and functions within each module |
| 4 — Code | [04-code.md](c4-diagrams/04-code.md) | Class diagram + hybrid search sequence diagram |

## Architecture Decision Records (ADRs)

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](adrs/ADR-001-langchain-for-llm-orchestration.md) | LangChain for LLM orchestration | Accepted |
| [ADR-002](adrs/ADR-002-neo4j-over-alternatives.md) | Neo4j as the graph database | Accepted |
| [ADR-003](adrs/ADR-003-in-memory-vector-store-for-prototyping.md) | In-memory vector store with mock embeddings | Accepted (temporary) |
| [ADR-004](adrs/ADR-004-parallel-hybrid-retrieval.md) | Parallel hybrid retrieval via ThreadPoolExecutor | Accepted |
| [ADR-005](adrs/ADR-005-pydantic-v2-domain-models.md) | Pydantic v2 for domain models | Accepted |
