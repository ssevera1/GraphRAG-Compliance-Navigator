# ADR-002: Neo4j as the Graph Database

**Status:** Accepted
**Date:** 2026-02-16

## Context

The compliance engine stores entities and relationships as a knowledge graph. We need a graph database that supports labeled property graphs, Cypher queries, and scales to millions of triplets.

## Decision

Use **Neo4j** with the official `neo4j` Python driver, connecting over the Bolt protocol.

## Rationale

- **Labeled property graph model**: Directly maps to our domain — nodes have labels (`Regulation`, `Company`, `Clause`) and relationships have types (`VIOLATES`, `COMPLIES_WITH`, `REQUIRES`).
- **Cypher query language**: Expressive pattern matching for graph traversals (e.g. finding all clauses that violate a regulation).
- **Mature ecosystem**: APOC plugins, graph data science library, and strong community support.
- **MERGE semantics**: Idempotent upserts prevent duplicate nodes/edges during repeated ingestion runs.

## Trade-offs

| Pro | Con |
|-----|-----|
| Purpose-built for graph traversals | Requires running a separate database service |
| Rich Cypher query language | Learning curve for developers new to graph DBs |
| ACID transactions | Heavier resource footprint than embedded alternatives |
| Scales to enterprise workloads | Community edition has clustering limitations |

## Alternatives Considered

- **NetworkX** (in-memory): Zero infrastructure, but no persistence and won't scale beyond a single machine's RAM.
- **Amazon Neptune**: Managed service, but vendor lock-in and higher cost for development.
- **ArangoDB**: Multi-model (document + graph), but weaker graph query ergonomics compared to Cypher.
