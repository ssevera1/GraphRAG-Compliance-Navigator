"""Neo4j-backed knowledge graph for compliance triplets."""

from __future__ import annotations

import logging
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.ingestion.extractor import Entity, ExtractionResult, Relationship

logger = logging.getLogger(__name__)


class KnowledgeGraphError(Exception):
    """Base exception for knowledge graph operations."""
    pass


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
        except AuthError as e:
            logger.error(f"Neo4j authentication failed for {uri}: {e}")
            raise KnowledgeGraphError(f"Authentication failed: {e}") from e
        except ServiceUnavailable as e:
            logger.error(f"Neo4j unavailable at {uri}: {e}")
            raise KnowledgeGraphError(f"Service unavailable: {e}") from e

    # ── public API ────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j connection")

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
                logger.debug(f"Added entity: {entity.type.value} '{entity.name}'")
        except Exception as e:
            logger.error(f"Failed to add entity '{entity.name}': {e}")
            raise KnowledgeGraphError(f"Failed to add entity: {e}") from e

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
                result = session.run(
                    query, src=rel.source, tgt=rel.target, props=rel.properties,
                )
                summary = result.consume()
                if summary.relationships_created == 0 and summary.relationships_deleted == 0:
                    logger.warning(
                        f"Relationship not created: nodes '{rel.source}' or '{rel.target}' may not exist"
                    )
                else:
                    logger.debug(f"Added relationship: {rel.source} -{rel.type.value}-> {rel.target}")
        except Exception as e:
            logger.error(f"Failed to add relationship {rel.source} -> {rel.target}: {e}")
            raise KnowledgeGraphError(f"Failed to add relationship: {e}") from e

    def add_extraction(self, result: ExtractionResult) -> None:
        """Persist every entity and relationship from an extraction run."""
        if not result.entities:
            logger.warning("Extraction result contains no entities")
        if not result.relationships:
            logger.warning("Extraction result contains no relationships")

        for entity in result.entities:
            self.add_entity(entity)
        for rel in result.relationships:
            self.add_relationship(rel)

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
                results = [record.data() for record in records]
                logger.debug(f"Found {len(results)} neighbours for '{entity_name}'")
                return results
        except Exception as e:
            logger.error(f"Failed to get neighbours for '{entity_name}': {e}")
            raise KnowledgeGraphError(f"Failed to query graph: {e}") from e

    def query(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as dicts."""
        if not cypher or not cypher.strip():
            logger.warning("query called with empty Cypher statement")
            return []

        try:
            with self._driver.session() as session:
                records = session.run(cypher, **params)
                results = [record.data() for record in records]
                logger.debug(f"Cypher query returned {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            raise KnowledgeGraphError(f"Query execution failed: {e}") from e

    def clear(self) -> None:
        """Delete all nodes and relationships (use with care)."""
        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared all nodes and relationships from graph")
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            raise KnowledgeGraphError(f"Failed to clear graph: {e}") from e
