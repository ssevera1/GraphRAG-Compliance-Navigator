# C4 Level 1 — System Context Diagram

Shows the GraphRAG Compliance Engine and its external actors.

```mermaid
C4Context
    title System Context — GraphRAG Compliance Engine

    Person(analyst, "Compliance Analyst", "Queries the system for regulation violations and compliance status")
    System(engine, "GraphRAG Compliance Engine", "Extracts entities/relationships from legal text, stores them in a knowledge graph, and answers compliance queries via hybrid retrieval")
    System_Ext(llm, "LLM Provider", "OpenAI / Anthropic / local model — performs entity extraction")
    SystemDb_Ext(neo4j, "Neo4j", "Graph database storing compliance triplets")

    Rel(analyst, engine, "Submits legal text & queries", "HTTP / CLI")
    Rel(engine, llm, "Sends extraction prompts", "HTTPS / API")
    Rel(engine, neo4j, "Reads/writes triplets", "Bolt")
    Rel(engine, analyst, "Returns compliance insights", "JSON")
```

## Data Flow Summary

| Flow | Latency Target | Notes |
|------|---------------|-------|
| Analyst → Engine (query) | < 200ms | Local parsing + parallel retrieval |
| Engine → LLM (extraction) | 1–5s | Depends on provider; batched where possible |
| Engine → Neo4j (read/write) | < 50ms | Local instance on Bolt protocol |
| Engine → Analyst (response) | < 2s total | Dominated by LLM call during ingestion; retrieval is fast |
