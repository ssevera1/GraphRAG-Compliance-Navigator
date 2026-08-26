"""Tests for LLM entity/relationship extraction.

The chat model is mocked throughout, so these run without an API key.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from src.ingestion.extractor import (
    EdgeType,
    ExtractionResult,
    NodeType,
    extract_entities_and_relationships,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "entities": [
        {"name": "Acme Corp", "type": "Company", "properties": {}},
        {"name": "GDPR", "type": "Regulation", "properties": {}},
    ],
    "relationships": [
        {
            "source": "Acme Corp",
            "target": "GDPR",
            "type": "VIOLATES",
            "properties": {},
        }
    ],
}


def _llm_returning(content):
    """A chat model whose invoke() returns a message with *content*."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def _llm_raising(exc):
    """A chat model whose invoke() raises *exc*."""
    llm = MagicMock()
    llm.invoke.side_effect = exc
    return llm


# ── Happy path ───────────────────────────────────────────────────────────────

class TestSuccessfulExtraction:

    def test_plain_json_is_parsed(self):
        result = extract_entities_and_relationships(
            "Acme Corp violated GDPR.", _llm_returning(json.dumps(VALID_PAYLOAD))
        )

        assert [e.name for e in result.entities] == ["Acme Corp", "GDPR"]
        assert result.entities[0].type is NodeType.COMPANY
        assert result.relationships[0].type is EdgeType.VIOLATES

    def test_markdown_fenced_json_is_parsed(self):
        fenced = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
        result = extract_entities_and_relationships("text", _llm_returning(fenced))

        assert len(result.entities) == 2

    def test_content_block_list_is_flattened(self):
        blocks = [{"type": "text", "text": json.dumps(VALID_PAYLOAD)}]
        result = extract_entities_and_relationships("text", _llm_returning(blocks))

        assert len(result.relationships) == 1


# ── Empty / unparseable model output ─────────────────────────────────────────

class TestEmptyResults:
    """A model that answered but said nothing usable is a legitimate empty."""

    @pytest.mark.parametrize("content", [None, "", "   ", "```json\n\n```"])
    def test_no_usable_content_returns_empty(self, content):
        result = extract_entities_and_relationships("text", _llm_returning(content))

        assert result == ExtractionResult()

    def test_unparseable_json_is_logged_and_returns_empty(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.ingestion.extractor"):
            result = extract_entities_and_relationships(
                "text", _llm_returning("not json at all")
            )

        assert result == ExtractionResult()
        assert "Failed to parse LLM response as JSON" in caplog.text
        # exc_info=True keeps the decoder's position information.
        assert any(r.exc_info for r in caplog.records)

    def test_schema_violation_returns_empty(self):
        payload = {"entities": [{"name": "Acme Corp", "type": "NotAType"}]}
        result = extract_entities_and_relationships(
            "text", _llm_returning(json.dumps(payload))
        )

        assert result == ExtractionResult()


# ── Invocation failures ──────────────────────────────────────────────────────

class TestInvocationFailure:
    """The regression this PR is about: a failed call must not look empty.

    Callers pass the result straight to KnowledgeGraph.add_extraction, so
    returning ExtractionResult() here would write a silently incomplete graph
    on a rate limit or an expired key.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("upstream timed out"),
            RuntimeError("429 rate limit exceeded"),
            PermissionError("401 invalid api key"),
            AttributeError("'dict' object has no attribute 'invoke'"),
        ],
    )
    def test_invoke_failure_propagates(self, exc):
        with pytest.raises(type(exc)):
            extract_entities_and_relationships("text", _llm_raising(exc))

    def test_invoke_failure_is_logged_with_traceback(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.ingestion.extractor"):
            with pytest.raises(RuntimeError):
                extract_entities_and_relationships(
                    "a" * 42, _llm_raising(RuntimeError("429 rate limit exceeded"))
                )

        records = [r for r in caplog.records if "LLM invocation failed" in r.message]
        assert len(records) == 1
        assert records[0].exc_info is not None
        assert "42 chars" in caplog.text
        # The log must not assert a cause it has not established.
        assert "timeout, rate limit, or connection error" not in caplog.text
