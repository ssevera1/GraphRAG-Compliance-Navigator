# ADR-004: Parallel Hybrid Retrieval via ThreadPoolExecutor

**Status:** Accepted
**Date:** 2026-02-16

## Context

Hybrid search combines two independent retrieval paths: vector similarity and graph traversal. These have no data dependency on each other — the graph arm needs entity names extracted from the query, and the vector arm needs the query embedding. Both can run concurrently.

## Decision

Use **`concurrent.futures.ThreadPoolExecutor`** with 2 workers to run the vector and graph search arms in parallel within `hybrid_search()`.

## Rationale

- **Latency reduction**: Both arms execute simultaneously. Total latency = max(vector_time, graph_time) instead of sum.
- **Simple concurrency model**: `ThreadPoolExecutor` is stdlib, no async framework required, and the two-task workload doesn't warrant heavier machinery.
- **I/O-bound workloads**: Both arms are I/O-bound (network calls to vector store / Neo4j), so threads are effective despite Python's GIL.

## Trade-offs

| Pro | Con |
|-----|-----|
| ~2x latency improvement for retrieval | Thread overhead for very fast lookups (< 1ms) is wasteful |
| No async framework dependency | Error handling is slightly more complex (futures) |
| Stdlib only, no new deps | Limited to 2 workers — not a general-purpose concurrency solution |

## Alternatives Considered

- **Sequential execution**: Simpler code, but doubles retrieval latency.
- **asyncio**: More Pythonic for I/O concurrency, but requires async Neo4j driver and async-compatible vector store — premature given the current prototype scope.
- **multiprocessing**: Overkill for 2 I/O-bound tasks; process startup overhead would negate gains.
