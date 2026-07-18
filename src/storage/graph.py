"""Neo4j-backed knowledge graph for compliance triplets."""

from __future__ import annotations

import logging
from typing import Optional

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.ingestion.extractor import Entity, ExtractionResult, Relationship

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Thin wrapper around a Neo4j instance for storing compliance triplets.

    Usage::

        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
        kg.add_extraction(result)
        neighbours = kg.get_neighbours("GDPR")
        kg.close()
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            self._driver: Driver = GraphDatabase.driver(
                uri, auth=(user, password), connection_timeout=10.0
            )
            self._driver.verify_connectivity()
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j at {uri}: {e}")
            raise
        logger.info(f"Connected to Neo4j at {uri}")

    # ── public API ────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.debug("Neo4j connection closed")

    def add_entity(self, entity: Entity) -> None:
        """Merge a single entity node."""
        if not entity.name or not entity.name.strip():
            logger.warning("Skipping entity with empty name")
            return
        query = (
            f"MERGE (n:{entity.type.value} {{name: $name}}) "
            "SET n += $props"
        )
        try:
            with self._driver.session() as session:
                session.run(query, name=entity.name, props=entity.properties)
        except Exception as e:
            logger.error(f"Failed to add entity {entity.name}: {e}")
            raise

    def add_relationship(self, rel: Relationship) -> None:
        """Merge a relationship between two already-existing nodes."""
        if not rel.source or not rel.source.strip():
            logger.warning(f"Skipping relationship with empty source")
            return
        if not rel.target or not rel.target.strip():
            logger.warning(f"Skipping relationship with empty target")
            return
        query = (
            "MATCH (a {name: $src}), (b {name: $tgt}) "
            f"MERGE (a)-[r:{rel.type.value}]->(b) "
            "SET r += $props"
        )
        try:
            with self._driver.session() as session:
                session.run(
                    query, src=rel.source, tgt=rel.target, props=rel.properties,
                )
        except Exception as e:
            logger.error(
                f"Failed to add relationship {rel.source}-{rel.type.value}-{rel.target}: {e}"
            )
            raise

    def add_extraction(self, result: ExtractionResult) -> None:
        """Persist every entity and relationship from an extraction run."""
        if not result.entities and not result.relationships:
            logger.warning("Empty extraction result received")
            return
        for entity in result.entities:
            self.add_entity(entity)
        for rel in result.relationships:
            self.add_relationship(rel)
        logger.info(
            f"Added {len(result.entities)} entities and "
            f"{len(result.relationships)} relationships"
        )

    def get_neighbours(self, entity_name: str) -> list[dict]:
        """Return all directly connected nodes (any direction, any type)."""
        if not entity_name or not entity_name.strip():
            logger.warning("get_neighbours called with empty entity name")
            return []
        query = (
            "MATCH (n {name: $name})-[r]-(m) "
            "RETURN n.name AS source, type(r) AS relationship, "
            "       m.name AS target, labels(m) AS target_labels, "
            "       properties(r) AS rel_props"
        )
        try:
            with self._driver.session() as session:
                records = session.run(query, name=entity_name)
                return [record.data() for record in records]
        except Exception as e:
            logger.error(f"Failed to query neighbours for {entity_name}: {e}")
            return []

    def query(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as dicts."""
        if not cypher or not cypher.strip():
            logger.warning("query called with empty Cypher statement")
            return []
        try:
            with self._driver.session() as session:
                records = session.run(cypher, **params)
                return [record.data() for record in records]
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []

    def clear(self) -> None:
        """Delete all nodes and relationships (use with care)."""
        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Knowledge graph cleared")
        except Exception as e:
            logger.error(f"Failed to clear knowledge graph: {e}")
            raise
