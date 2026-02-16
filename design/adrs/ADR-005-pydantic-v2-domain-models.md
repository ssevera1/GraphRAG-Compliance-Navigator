# ADR-005: Pydantic v2 for Domain Models

**Status:** Accepted
**Date:** 2026-02-16

## Context

The extraction pipeline parses raw JSON from LLM responses into structured Python objects. We need validation, serialization, and clear schema definitions for entities and relationships.

## Decision

Use **Pydantic v2** (`BaseModel`, `model_validate()`) for all domain types: `Entity`, `Relationship`, and `ExtractionResult`.

## Rationale

- **Validation at the boundary**: LLM output is unpredictable — Pydantic catches malformed JSON, missing fields, and invalid enum values before they propagate.
- **Self-documenting schemas**: `Field(description=...)` serves as both runtime validation and API documentation.
- **Enum integration**: `NodeType` and `EdgeType` are `str, Enum` — Pydantic serializes them as strings for JSON and validates them as enums in Python.
- **Performance**: Pydantic v2's Rust core (`pydantic-core`) is significantly faster than v1 for validation-heavy workloads.

## Trade-offs

| Pro | Con |
|-----|-----|
| Catches LLM output errors early | Strict validation can reject "almost correct" LLM output |
| Clean serialization to/from JSON | Learning curve for v1 → v2 migration patterns |
| Fast Rust-backed validation | Adds a dependency (though already required by LangChain) |

## Alternatives Considered

- **dataclasses**: Lighter, stdlib, but no built-in validation or JSON schema generation.
- **attrs**: Good validation via `attrs-strict`, but less ecosystem integration with LangChain.
- **TypedDict**: Zero overhead, but no runtime validation — risky for LLM-generated data.
