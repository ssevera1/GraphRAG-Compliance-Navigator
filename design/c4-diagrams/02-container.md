# C4 Level 2 — Container Diagram

Shows the major containers (deployable units) within the engine.

```mermaid
C4Container
    title Container Diagram — GraphRAG Compliance Engine

    Person(analyst, "Compliance Analyst")

    Container_Boundary(engine, "GraphRAG Compliance Engine") {
        Container(ingestion, "Ingestion Service", "Python / LangChain", "Accepts raw legal text, calls LLM to extract entities & relationships")
        Container(storage, "Graph Storage", "Python / neo4j-driver", "Persists and queries compliance triplets in Neo4j")
        Container(retrieval, "Retrieval Service", "Python", "Hybrid search: vector similarity + graph traversal in parallel")
        ContainerDb(vectorstore, "Vector Store", "In-memory (mock)", "Stores document embeddings for similarity search")
    }

    SystemDb_Ext(neo4j, "Neo4j", "Graph database")
    System_Ext(llm, "LLM Provider", "OpenAI / Anthropic")

    Rel(analyst, retrieval, "Sends natural-language query")
    Rel(analyst, ingestion, "Submits legal text chunks")
    Rel(ingestion, llm, "Extraction prompts", "HTTPS")
    Rel(ingestion, storage, "ExtractionResult objects")
    Rel(storage, neo4j, "Cypher MERGE queries", "Bolt")
    Rel(retrieval, vectorstore, "Cosine similarity search")
    Rel(retrieval, storage, "get_neighbours()")
    Rel(storage, neo4j, "Cypher MATCH queries", "Bolt")
    Rel(retrieval, analyst, "HybridResult")
```
