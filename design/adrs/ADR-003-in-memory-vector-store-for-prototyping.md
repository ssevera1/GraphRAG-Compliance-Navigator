# ADR-003: In-Memory Vector Store with Mock Embeddings for Prototyping

**Status:** Accepted (temporary — to be replaced)
**Date:** 2026-02-16

## Context

The hybrid retrieval pipeline requires a vector similarity search arm. For the initial prototype, we need a working vector store without depending on external embedding services or vector databases.

## Decision

Use a **custom in-memory `VectorStore`** with deterministic MD5-based mock embeddings (64-dim vectors). Cosine similarity is computed directly in Python.

## Rationale

- **Zero external dependencies**: Tests and development run without API keys or vector DB infrastructure.
- **Deterministic**: MD5-based embeddings produce the same vector for the same input, making tests reproducible.
- **Swappable**: The `VectorStore` interface (`add()`, `search()`) is simple enough that replacing it with a real implementation (FAISS, Pinecone, Chroma) requires minimal changes.

## Trade-offs

| Pro | Con |
|-----|-----|
| No API keys needed for dev/test | Mock embeddings have no semantic meaning |
| Fully deterministic and reproducible | O(n) brute-force search, won't scale |
| Simple implementation (~30 lines) | Not suitable for production workloads |

## Migration Path

1. Replace `dummy_embed()` with a real embedding function (e.g. `OpenAIEmbeddings`, `sentence-transformers`).
2. Replace the in-memory store with a vector database (FAISS for local, Pinecone/Weaviate/Chroma for managed).
3. Update `VectorStore.search()` to use the database's native ANN search.
