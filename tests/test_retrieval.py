"""Tests for the hybrid retrieval pipeline.

These tests mock the Neo4j driver so they run without a live database.
"""

from __future__ import annotations

import logging
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
    DummyEmbeddings,
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
    vs = VectorStore(embedder=DummyEmbeddings())
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


class TestGraphTraversalObservability:
    """A malformed get_neighbours response must be visible, not silent.

    Without these, 'the entity genuinely has no neighbours' and 'the graph
    backend returned garbage' are indistinguishable at the call site.
    """

    def _kg_returning(self, value, only_for="GDPR"):
        """Return `value` for one entity; the extractor also yields "What"."""
        kg = MagicMock(spec=KnowledgeGraph)
        kg.get_neighbours = lambda name: value if name == only_for else []
        return kg

    def test_none_response_is_logged_and_skipped(self, caplog):
        vs = _build_vector_store()
        kg = self._kg_returning(None)

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            result = hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert result.graph_results == []
        assert "returned None" in caplog.text
        assert "GDPR" in caplog.text

    def test_non_iterable_response_is_logged_and_skipped(self, caplog):
        vs = _build_vector_store()
        kg = self._kg_returning(42)

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            result = hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert result.graph_results == []
        assert "returned int" in caplog.text

    def test_string_response_is_treated_as_invalid_not_iterated(self, caplog):
        """A str is iterable, but iterating it yields characters, not records."""
        vs = _build_vector_store()
        kg = self._kg_returning("not a record list")

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            result = hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert result.graph_results == []
        assert "returned str" in caplog.text

    def test_malformed_record_is_logged_and_the_rest_survive(self, caplog):
        """One bad record must not discard the good ones alongside it."""
        good = {
            "source": "GDPR",
            "relationship": "REQUIRES",
            "target": "Article 5",
            "target_labels": ["Clause"],
            "rel_props": {},
        }
        vs = _build_vector_store()
        kg = self._kg_returning([good, "junk", {}, None])

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            result = hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert result.graph_results == [good]
        assert "invalid record (str)" in caplog.text
        assert "invalid record (empty dict)" in caplog.text
        assert "invalid record (NoneType)" in caplog.text

    def test_generator_response_is_accepted(self):
        """A streamed cursor must not be discarded by an over-strict check."""
        good = {
            "source": "GDPR",
            "relationship": "REQUIRES",
            "target": "Article 5",
            "target_labels": ["Clause"],
            "rel_props": {},
        }
        vs = _build_vector_store()
        kg = MagicMock(spec=KnowledgeGraph)
        kg.get_neighbours = lambda name: (r for r in ([good] if name == "GDPR" else []))

        result = hybrid_search(
            query="What clauses violate GDPR?",
            vector_store=vs,
            knowledge_graph=kg,
        )

        assert result.graph_results == [good]

    def test_well_formed_response_logs_nothing(self, caplog):
        vs = _build_vector_store()
        kg = _mock_knowledge_graph()

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert caplog.text == ""


class TestHybridArmFailureIsVisible:
    """A whole arm collapsing must not look like a query with no matches."""

    def test_graph_arm_failure_is_logged_and_vector_survives(self, caplog):
        vs = _build_vector_store()
        kg = MagicMock(spec=KnowledgeGraph)

        def _boom(name):
            raise RuntimeError("neo4j is down")

        kg.get_neighbours = _boom

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            result = hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert result.graph_results == []
        assert len(result.vector_results) > 0, "the healthy arm must still answer"
        assert "graph arm of hybrid_search failed" in caplog.text
        assert "neo4j is down" in caplog.text

    def test_vector_arm_failure_is_logged_and_graph_survives(self, caplog):
        vs = MagicMock()
        vs.search.side_effect = RuntimeError("embedding backend unreachable")
        kg = _mock_knowledge_graph()

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            result = hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert result.vector_results == []
        assert len(result.graph_results) > 0, "the healthy arm must still answer"
        assert "vector arm of hybrid_search failed" in caplog.text

    def test_healthy_query_logs_no_arm_failure(self, caplog):
        vs = _build_vector_store()
        kg = _mock_knowledge_graph()

        with caplog.at_level(logging.WARNING, logger="src.retrieval.search"):
            hybrid_search(
                query="What clauses violate GDPR?",
                vector_store=vs,
                knowledge_graph=kg,
            )

        assert "hybrid_search failed" not in caplog.text
