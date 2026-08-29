"""Neo4j-backed knowledge graph for compliance triplets."""

from __future__ import annotations

import time
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import AuthError, ConfigurationError

from src.ingestion.extractor import Entity, ExtractionResult, Relationship

_MAX_RETRIES = 5
_BASE_DELAY = 0.5


def _connect_with_retries(uri: str, user: str, password: str) -> Driver:
    """Return a Driver whose connectivity has been verified.

    ``GraphDatabase.driver()`` only builds a connection pool; it does not talk
    to the server, so the failure surfaces from ``verify_connectivity()`` when
    the driver object already exists. Each failed attempt is therefore closed
    before the next one, otherwise a Neo4j that is still booting leaves one
    orphaned pool (and its background threads) per retry.

    Misconfiguration -- bad credentials or a bad URI -- is not transient, so it
    propagates immediately instead of stalling for the full retry budget.
    """
    for attempt in range(_MAX_RETRIES):
        driver: Driver | None = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            return driver
        except (AuthError, ConfigurationError):
            if driver is not None:
                driver.close()
            raise
        except Exception as exc:
            if driver is not None:
                driver.close()
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Failed to connect to Neo4j at {uri} "
                    f"after {_MAX_RETRIES} attempts"
                ) from exc
            time.sleep(_BASE_DELAY * (2 ** attempt))

    raise RuntimeError(  # pragma: no cover - _MAX_RETRIES is always >= 1
        f"Failed to connect to Neo4j at {uri}: no connection attempts were made"
    )


class KnowledgeGraph:
    """Thin wrapper around a Neo4j instance for storing compliance triplets.

    Usage::

        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
        kg.add_extraction(result)
        neighbours = kg.get_neighbours("GDPR")
        kg.close()
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: Driver = _connect_with_retries(uri, user, password)

    # ── public API ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()

    def add_entity(self, entity: Entity) -> None:
        """Merge a single entity node."""
        query = (
            f"MERGE (n:{entity.type.value} {{name: $name}}) "
            "SET n += $props"
        )
        with self._driver.session() as session:
            session.run(query, name=entity.name, props=entity.properties)

    def add_relationship(self, rel: Relationship) -> None:
        """Merge a relationship between two already-existing nodes."""
        query = (
            "MATCH (a {name: $src}), (b {name: $tgt}) "
            f"MERGE (a)-[r:{rel.type.value}]->(b) "
            "SET r += $props"
        )
        with self._driver.session() as session:
            session.run(
                query, src=rel.source, tgt=rel.target, props=rel.properties,
            )

    def add_extraction(self, result: ExtractionResult) -> None:
        """Persist every entity and relationship from an extraction run."""
        for entity in result.entities:
            self.add_entity(entity)
        for rel in result.relationships:
            self.add_relationship(rel)

    def get_neighbours(self, entity_name: str) -> list[dict]:
        """Return all directly connected nodes (any direction, any type)."""
        query = (
            "MATCH (n {name: $name})-[r]-(m) "
            "RETURN n.name AS source, type(r) AS relationship, "
            "       m.name AS target, labels(m) AS target_labels, "
            "       properties(r) AS rel_props"
        )
        with self._driver.session() as session:
            records = session.run(query, name=entity_name)
            return [record.data() for record in records]

    def query(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as dicts."""
        with self._driver.session() as session:
            records = session.run(cypher, **params)
            return [record.data() for record in records]

    def clear(self) -> None:
        """Delete all nodes and relationships (use with care)."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
