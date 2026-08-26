"""Entity and relationship extraction from legal text using an LLM."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
) -> ExtractionResult:
    """Send *text* to the LLM and parse structured entities / relationships.

    Parameters
    ----------
    text:
        Raw legal text chunk to analyse.
    llm:
        Any LangChain chat model (OpenAI, Anthropic, local, …).

    Returns
    -------
    ExtractionResult
        Parsed entities and relationships. Empty if the model returned no
        content or content this function could not parse as the JSON schema.

    Raises
    ------
    Exception
        Whatever ``llm.invoke`` raises. An extraction that never reached the
        model is deliberately *not* reported as an empty result: callers feed
        this straight into ``KnowledgeGraph.add_extraction``, so swallowing the
        failure would write a silently incomplete graph.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract entities and relationships from:\n\n{text}"),
    ]

    try:
        response = llm.invoke(messages)
    except Exception:
        # Logged here because this is where the chunk being extracted is known,
        # then re-raised: only the caller driving the ingestion loop can decide
        # whether to skip the chunk, retry, or abort the batch.
        logger.warning(
            "LLM invocation failed for a chunk of %d chars",
            len(text),
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
