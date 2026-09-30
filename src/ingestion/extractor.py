"""Entity and relationship extraction from legal text using an LLM."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional
import time
import random

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Failures that will not succeed on retry: an expired/invalid credential, or a
# caller bug (wrong argument type, unexpected response shape). Retrying these
# only stalls the ingestion run for the full backoff budget before failing anyway.
_NON_TRANSIENT_EXCEPTIONS = (PermissionError, AttributeError, TypeError)


# ── Domain types ──────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    REGULATION = "Regulation"
    COMPANY = "Company"
    CLAUSE = "Clause"


class EdgeType(str, Enum):
    VIOLATES = "VIOLATES"
    COMPLIES_WITH = "COMPLIES_WITH"
    REQUIRES = "REQUIRES"


class Entity(BaseModel):
    """A node extracted from legal text."""
    name: str = Field(description="Canonical name of the entity")
    type: NodeType = Field(description="Category of the entity")
    properties: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    """A directed edge between two entities."""
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    type: EdgeType = Field(description="Kind of relationship")
    properties: dict = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Complete extraction output for a text chunk."""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a legal-document analysis assistant.
Given a text chunk, extract structured entities and relationships.

Entity types (Nodes):
  - Regulation  : A law, regulation, or standard (e.g. "GDPR", "SOX").
  - Company     : An organisation mentioned in the text.
  - Clause      : A specific clause, article, or section of a regulation.

Relationship types (Edges):
  - VIOLATES      : source entity violates the target regulation/clause.
  - COMPLIES_WITH : source entity complies with the target regulation/clause.
  - REQUIRES      : a regulation/clause requires something of the target entity.

Return ONLY valid JSON matching this schema (no extra keys):
{
  "entities": [
    {"name": "...", "type": "Regulation|Company|Clause", "properties": {}}
  ],
  "relationships": [
    {"source": "...", "target": "...", "type": "VIOLATES|COMPLIES_WITH|REQUIRES", "properties": {}}
  ]
}
"""


# ── Extraction function ──────────────────────────────────────────────────────

def _as_text(content: str | list[str | dict[Any, Any]]) -> str:
    """Flatten a message payload to text.

    A chat model returns either a plain string or a list of content blocks;
    treating the list case as a string raises AttributeError at runtime.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def extract_entities_and_relationships(
    text: str,
    llm: BaseChatModel,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> ExtractionResult:
    """Send *text* to the LLM and parse structured entities / relationships.

    Parameters
    ----------
    text:
        Raw legal text chunk to analyse.
    llm:
        Any LangChain chat model (OpenAI, Anthropic, local, …).
    max_retries:
        Maximum number of retry attempts for transient failures (default: 3).
    initial_delay:
        Initial delay in seconds before first retry (default: 1.0).

    Returns
    -------
    ExtractionResult
        Parsed entities and relationships. Empty if the model returned no
        content or content this function could not parse as the JSON schema.

    Raises
    ------
    Exception
        Whatever ``llm.invoke`` raises after exhausting retries. An extraction
        that never reached the model is deliberately *not* reported as an empty
        result: callers feed this straight into ``KnowledgeGraph.add_extraction``,
        so swallowing the failure would write a silently incomplete graph.
    ValueError
        If ``max_retries`` is negative.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract entities and relationships from:\n\n{text}"),
    ]

    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(messages)
            break
        except _NON_TRANSIENT_EXCEPTIONS:
            # Authentication failures and programming bugs will not succeed on
            # retry, so burning the full backoff budget only delays a failure
            # that is already certain. See src/storage/graph.py for the same
            # distinction on the Neo4j connection path.
            logger.warning(
                "LLM invocation failed for a chunk of %d chars (non-transient)",
                len(text),
                exc_info=True,
            )
            raise
        except Exception:
            if attempt < max_retries:
                jitter = random.uniform(0, 0.1 * delay)
                wait_time = delay + jitter
                logger.debug(
                    "LLM invocation failed (attempt %d/%d), retrying in %.2f seconds",
                    attempt + 1,
                    max_retries + 1,
                    wait_time,
                )
                time.sleep(wait_time)
                delay *= 2
            else:
                logger.warning(
                    "LLM invocation failed for a chunk of %d chars after %d attempts",
                    len(text),
                    max_retries + 1,
                    exc_info=True,
                )
                raise

    if response.content is None:
        logger.debug("LLM returned empty content")
        return ExtractionResult()

    content = _as_text(response.content)

    if not content or not content.strip():
        logger.debug("LLM response content is empty after text extraction")
        return ExtractionResult()

    # Strip markdown fences if the model wraps the JSON.
    if "```" in content:
        content = content.split("```json")[-1].split("```")[0]

    content = content.strip()
    if not content:
        logger.debug("LLM response is empty after markdown fence removal")
        return ExtractionResult()

    try:
        parsed = json.loads(content)
        return ExtractionResult.model_validate(parsed)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse LLM response as JSON", exc_info=True)
        return ExtractionResult()
