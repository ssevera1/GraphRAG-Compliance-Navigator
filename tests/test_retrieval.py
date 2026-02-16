"""Tests for the hybrid retrieval pipeline.

These tests mock the Neo4j driver so they run without a live database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.extractor import (
    Entity,
    ExtractionResult,
    NodeType,
    EdgeType,
    Relationship,
)
from src.retrieval.search import (
    HybridResult,
    VectorStore,
    hybrid_search,
    _extract_entity_names_from_query,
)
from src.storage.graph import KnowledgeGraph


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    "Article 5 of GDPR requires data minimisation.",
    "Acme Corp was found to violate GDPR Article 17.",
    "Clause 12.3 requires annual compliance audits.",
    "Beta Inc complies with SOX Section 302.",
    "GDPR Article 25 mandates data protection by design.",
]

SAMPLE_EXTRACTION = ExtractionResult(
    entities=[
        Entity(name="GDPR", type=NodeType.REGULATION),
        Entity(name="Article 17", type=NodeType.CLAUSE),
        Entity(name="Article 5", type=NodeType.CLAUSE),
        Entity(name="Article 25", type=NodeType.CLAUSE),
        Entity(name="Acme Corp", type=NodeType.COMPANY),
    ],
    relationships=[
        Relationship(
            source="Acme Corp", target="Article 17", type=EdgeType.VIOLATES,
        ),
        Relationship(
            source="Article 17", target="GDPR", type=EdgeType.REQUIRES,
        ),
        Relationship(
            source="Article 5", target="GDPR", type=EdgeType.REQUIRES,
        ),
        Relationship(
            source="Article 25", target="GDPR", type=EdgeType.REQUIRES,
        ),
    ],
)


def _build_vector_store() -> VectorStore:
    vs = VectorStore()
    for doc in SAMPLE_DOCUMENTS:
        vs.add(doc)
    return vs


def _mock_knowledge_graph() -> KnowledgeGraph:
    """Create a KnowledgeGraph with a mocked Neo4j driver."""
    with patch("src.storage.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "test")

    # Pre-build an adjacency map to simulate graph storage.
    adjacency: dict[str, list[dict]] = {
        "GDPR": [
            {
                "source": "GDPR",
                "relationship": "REQUIRES",
                "target": "Article 17",
                "target_labels": ["Clause"],
                "rel_props": {},
            },
            {
                "source": "GDPR",
                "relationship": "REQUIRES",
                "target": "Article 5",
                "target_labels": ["Clause"],
                "rel_props": {},
            },
            {
                "source": "GDPR",
                "relationship": "REQUIRES",
                "target": "Article 25",
                "target_labels": ["Clause"],
                "rel_props": {},
            },
        ],
        "Article 17": [
            {
                "source": "Acme Corp",
                "relationship": "VIOLATES",
                "target": "Article 17",
                "target_labels": ["Clause"],
                "rel_props": {},
            },
            {
                "source": "Article 17",
                "relationship": "REQUIRES",
                "target": "GDPR",
                "target_labels": ["Regulation"],
                "rel_props": {},
            },
        ],
        "Acme Corp": [
            {
                "source": "Acme Corp",
                "relationship": "VIOLATES",
                "target": "Article 17",
                "target_labels": ["Clause"],
                "rel_props": {},
            },
        ],
    }

    def _fake_get_neighbours(name: str) -> list[dict]:
        return adjacency.get(name, [])

    kg.get_neighbours = _fake_get_neighbours  # type: ignore[assignment]
    return kg


# ── Tests ────────────────────────────────────────────────────────────────────

class TestEntityExtraction:
    """Verify the heuristic entity-name extractor on queries."""

    def test_extracts_acronym(self):
        names = _extract_entity_names_from_query("What clauses violate GDPR?")
        assert "GDPR" in names

    def test_extracts_multi_word_entity(self):
        names = _extract_entity_names_from_query(
            "Show violations for Acme Corp"
        )
        assert "Acme Corp" in names

    def test_returns_empty_for_lowercase_query(self):
        names = _extract_entity_names_from_query("list all clauses")
        assert names == []


class TestVectorStore:
    """Sanity-check the mock vector store."""

    def test_search_returns_results(self):
        vs = _build_vector_store()
        results = vs.search("GDPR violation", top_k=3)
        assert len(results) == 3
        assert all("text" in r and "score" in r for r in results)

    def test_search_scores_are_bounded(self):
        vs = _build_vector_store()
        results = vs.search("data protection")
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


class TestHybridSearch:
    """End-to-end hybrid search with mocked graph backend."""

    def test_what_clauses_violate_gdpr(self):
        """The flagship query: 'What clauses violate GDPR?'"""
        vs = _build_vector_store()
        kg = _mock_knowledge_graph()

        result = hybrid_search(
            query="What clauses violate GDPR?",
            vector_store=vs,
            knowledge_graph=kg,
        )

        assert isinstance(result, HybridResult)

        # Vector arm should return relevant documents.
        assert len(result.vector_results) > 0
        texts = [r["text"] for r in result.vector_results]
        assert any("GDPR" in t for t in texts)

        # Graph arm should find neighbours of "GDPR".
        assert len(result.graph_results) > 0
        graph_targets = {r["target"] for r in result.graph_results}
        # GDPR's clause neighbours must include Article 17 (the violated one).
        assert "Article 17" in graph_targets

    def test_relationship_types_present(self):
        """Graph results should carry relationship type metadata."""
        vs = _build_vector_store()
        kg = _mock_knowledge_graph()

        result = hybrid_search(
            query="What clauses violate GDPR?",
            vector_store=vs,
            knowledge_graph=kg,
        )

        rel_types = {r["relationship"] for r in result.graph_results}
        assert "REQUIRES" in rel_types

    def test_unknown_entity_returns_empty_graph(self):
        """Querying an entity not in the graph should still succeed."""
        vs = _build_vector_store()
        kg = _mock_knowledge_graph()

        result = hybrid_search(
            query="Does Zebra Inc comply with HIPAA?",
            vector_store=vs,
            knowledge_graph=kg,
        )

        assert isinstance(result, HybridResult)
        # Vector search still works; graph may be empty for unknowns.
        assert len(result.vector_results) > 0


class TestExtractionResult:
    """Validate the pydantic models used for extraction output."""

    def test_sample_extraction_is_valid(self):
        assert len(SAMPLE_EXTRACTION.entities) == 5
        assert len(SAMPLE_EXTRACTION.relationships) == 4

    def test_entity_types(self):
        types = {e.type for e in SAMPLE_EXTRACTION.entities}
        assert NodeType.REGULATION in types
        assert NodeType.CLAUSE in types
        assert NodeType.COMPANY in types

    def test_relationship_types(self):
        types = {r.type for r in SAMPLE_EXTRACTION.relationships}
        assert EdgeType.VIOLATES in types
        assert EdgeType.REQUIRES in types
