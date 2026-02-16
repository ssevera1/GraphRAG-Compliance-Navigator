# GraphRAG Compliance Engine — Project Context

## Purpose
Legal document analysis engine that uses GraphRAG (Graph-based Retrieval-Augmented Generation) to extract compliance entities and relationships from legal text, store them in a knowledge graph, and retrieve them via hybrid search.

## Architecture

```
src/
├── ingestion/       # Text → structured data (LLM extraction)
│   └── extractor.py
├── storage/         # Persistence layer (Neo4j knowledge graph)
│   └── graph.py
└── retrieval/       # Query layer (hybrid vector + graph search)
    └── search.py
tests/
└── test_retrieval.py
```

## Domain Model

### Node types (entities)
- **Regulation** — A law, regulation, or standard (e.g. GDPR, SOX)
- **Company** — An organisation mentioned in legal text
- **Clause** — A specific clause, article, or section of a regulation

### Edge types (relationships)
- **VIOLATES** — Source entity violates the target regulation/clause
- **COMPLIES_WITH** — Source entity complies with the target regulation/clause
- **REQUIRES** — A regulation/clause requires something of the target entity

## Key Design Decisions

1. **LLM-agnostic extraction**: `extract_entities_and_relationships()` accepts any LangChain `BaseChatModel` — not coupled to a specific provider.
2. **Pydantic v2 models**: All domain types (`Entity`, `Relationship`, `ExtractionResult`) use Pydantic v2 `BaseModel` with `model_validate()`.
3. **Neo4j via official driver**: `KnowledgeGraph` uses `neo4j.GraphDatabase.driver()` directly (not an OGM). Cypher queries use `MERGE` for idempotent upserts.
4. **Hybrid retrieval runs in parallel**: `hybrid_search()` uses `ThreadPoolExecutor` to run vector similarity and graph traversal concurrently, returning a `HybridResult` with both result sets.
5. **Mock embeddings for now**: `VectorStore` uses deterministic MD5-based 64-dim embeddings. Swap for a real embedding model (e.g. OpenAI, sentence-transformers) when ready.
6. **Tests mock Neo4j**: All tests run without a live database. The driver is patched, and `get_neighbours` is replaced with an in-memory adjacency map.

## Conventions

- **Python ≥ 3.10**, using `from __future__ import annotations` for modern type hints
- **Imports**: Use `src.` prefix for internal imports (package is installed editable via `pip install -e .`)
- **Enums**: Node and edge types are `str, Enum` subclasses for type safety + JSON serialization
- **Testing**: pytest with `unittest.mock`. Tests are in `tests/` and mirrored by module name (e.g. `test_retrieval.py`)
- **No hardcoded credentials**: Neo4j URI/user/password are constructor parameters, never defaults

## Dependencies
- `langchain` + `langchain-openai` — LLM orchestration
- `neo4j` — Graph database driver
- `pydantic` ≥ 2.0 — Data validation
- `pytest` + `pytest-asyncio` — Testing (dev)

## Commands
- Install: `pip install -e ".[dev]"`
- Test: `python -m pytest tests/ -v`
