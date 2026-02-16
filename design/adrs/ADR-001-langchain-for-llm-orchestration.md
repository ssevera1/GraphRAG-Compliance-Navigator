# ADR-001: LangChain for LLM Orchestration

**Status:** Accepted
**Date:** 2026-02-16

## Context

The extraction pipeline needs to call an LLM with structured prompts and parse JSON responses. We need to support multiple LLM providers (OpenAI, Anthropic, local models) without rewriting the extraction logic for each one.

## Decision

Use **LangChain** (`langchain-core` + provider packages like `langchain-openai`) as the LLM abstraction layer.

## Rationale

- **Provider agnosticism**: `BaseChatModel` lets us swap OpenAI for Anthropic or a local model with zero changes to `extractor.py`.
- **Ecosystem**: LangChain provides built-in message types (`SystemMessage`, `HumanMessage`), output parsers, and chain composition if the pipeline grows.
- **Community adoption**: Widely used, well-documented, active maintenance.

## Trade-offs

| Pro | Con |
|-----|-----|
| Swap providers with a single line change | Adds a heavy dependency tree (~20 transitive packages) |
| Rich ecosystem of tools and integrations | Frequent breaking changes between major versions |
| Structured message API | Abstraction overhead for simple prompt → JSON workflows |

## Alternatives Considered

- **Direct API calls** (e.g. `openai` SDK): Simpler, fewer deps, but locks us to one provider.
- **LiteLLM**: Lighter abstraction, but smaller ecosystem and less mature tooling.
