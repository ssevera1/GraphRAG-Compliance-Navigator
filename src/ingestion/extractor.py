"""Entity and relationship extraction from legal text using an LLM."""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


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
        Parsed entities and relationships.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract entities and relationships from:\n\n{text}"),
    ]

    response = llm.invoke(messages)
    content = response.content

    # Strip markdown fences if the model wraps the JSON.
    if "```" in content:
        content = content.split("```json")[-1].split("```")[0]

    parsed = json.loads(content)
    return ExtractionResult.model_validate(parsed)
