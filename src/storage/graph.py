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
            self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j at {uri}: {e}")
            raise

    # ── public API ────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.debug("Neo4j driver closed")

    def add_entity(self, entity: Optional[Entity]) -> None:
        """Merge a single entity node."""
        if entity is None:
            logger.warning("Attempted to add None entity, skipping")
            return
        if not entity.name or not entity.name.strip():
            logger.warning("Attempted to add entity with empty name, skipping")
            return

        query = (
            f"MERGE (n:{entity.type.value} {{name: $name}}) "
            "SET n += $props"
        )
        try:
            with self._driver.session() as session:
                session.run(query, name=entity.name, props=entity.properties)
            logger.debug(f"Added entity: {entity.name} ({entity.type.value})")
        except Exception as e:
            logger.error(f"Failed to add entity {entity.name}: {e}")
            raise

    def add_relationship(self, rel: Optional[Relationship]) -> None:
        """Merge a relationship between two already-existing nodes."""
        if rel is None:
            logger.warning("Attempted to add None relationship, skipping")
            return
        if not rel.source or not rel.source.strip() or not rel.target or not rel.target.strip():
            logger.warning(
                f"Attempted to add relationship with empty source/target: "
                f"{rel.source} -> {rel.target}, skipping"
            )
            return

        query = (
            "MATCH (a {name: $src}), (b {name: $tgt}) "
            f"MERGE (a)-[r:{rel.type.value}]->(b) "
            "SET r += $props"
        )
        try:
            with self._driver.session() as session:
                result = session.run(
                    query, src=rel.source, tgt=rel.target, props=rel.properties,
                )
                summary = result.consume()
                if summary.counters.relationships_created == 0 and summary.counters.relationships_updated == 0:
                    logger.warning(
                        f"Relationship not created (nodes may not exist): "
                        f"{rel.source} -[{rel.type.value}]-> {rel.target}"
                    )
            logger.debug(f"Added relationship: {rel.source} -[{rel.type.value}]-> {rel.target}")
        except Exception as e:
            logger.error(
                f"Failed to add relationship {rel.source} -> {rel.target}: {e}"
            )
            raise

    def add_extraction(self, result: Optional[ExtractionResult]) -> None:
        """Persist every entity and relationship from an extraction run."""
        if result is None:
            logger.warning("Attempted to add None extraction result, skipping")
            return

        if not result.entities and not result.relationships:
            logger.warning("Extraction result is empty (no entities or relationships)")

        for entity in result.entities:
            self.add_entity(entity)
        for rel in result.relationships:
            self.add_relationship(rel)

        logger.info(
            f"Added extraction: {len(result.entities)} entities, "
            f"{len(result.relationships)} relationships"
        )

    def get_neighbours(self, entity_name: Optional[str]) -> list[dict]:
        """Return all directly connected nodes (any direction, any type)."""
        if not entity_name or not entity_name.strip():
            logger.warning("get_neighbours called with empty entity_name")
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
                result = [record.data() for record in records]
            logger.debug(f"Found {len(result)} neighbours for {entity_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to get neighbours for {entity_name}: {e}")
            return []

    def query(self, cypher: Optional[str], **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as dicts."""
        if not cypher or not cypher.strip():
            logger.warning("query called with empty Cypher statement")
            return []

        try:
            with self._driver.session() as session:
                records = session.run(cypher, **params)
                result = [record.data() for record in records]
            logger.debug(f"Executed query, got {len(result)} results")
            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []

    def clear(self) -> None:
        """Delete all nodes and relationships (use with care)."""
        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared all nodes and relationships from graph")
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            raise
