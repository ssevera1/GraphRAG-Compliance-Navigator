"""Neo4j-backed knowledge graph for compliance triplets."""

from __future__ import annotations

import logging
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, DriverError

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
        except (ServiceUnavailable, DriverError) as e:
            logger.error(f"Failed to connect to Neo4j at {uri}: {e}")
            raise

    # ── public API ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()
        logger.debug("Neo4j driver closed")

    def add_entity(self, entity: Entity) -> None:
        """Merge a single entity node."""
        if not entity.name or not entity.name.strip():
            logger.warning("Attempted to add entity with empty name")
            return

        query = (
            f"MERGE (n:{entity.type.value} {{name: $name}}) "
            "SET n += $props"
        )
        try:
            with self._driver.session() as session:
                session.run(query, name=entity.name, props=entity.properties)
                logger.debug(f"Added entity: {entity.name} ({entity.type.value})")
        except DriverError as e:
            logger.error(f"Failed to add entity {entity.name}: {e}")
            raise

    def add_relationship(self, rel: Relationship) -> None:
        """Merge a relationship between two already-existing nodes."""
        if not rel.source or not rel.source.strip():
            logger.warning("Attempted to add relationship with empty source")
            return
        if not rel.target or not rel.target.strip():
            logger.warning("Attempted to add relationship with empty target")
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
                if result.consume().counters.relationships_created == 0:
                    logger.warning(
                        f"No relationship created: {rel.source} -[{rel.type.value}]-> {rel.target} "
                        "(nodes may not exist)"
                    )
                else:
                    logger.debug(
                        f"Added relationship: {rel.source} -[{rel.type.value}]-> {rel.target}"
                    )
        except DriverError as e:
            logger.error(
                f"Failed to add relationship {rel.source} -> {rel.target}: {e}"
            )
            raise

    def add_extraction(self, result: ExtractionResult) -> None:
        """Persist every entity and relationship from an extraction run."""
        if not result.entities and not result.relationships:
            logger.warning("add_extraction called with empty result")
            return

        logger.info(
            f"Adding extraction: {len(result.entities)} entities, "
            f"{len(result.relationships)} relationships"
        )
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
                logger.debug(f"Found {len(results)} neighbours for {entity_name}")
                return results
        except DriverError as e:
            logger.error(f"Failed to query neighbours for {entity_name}: {e}")
            raise

    def query(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as dicts."""
        try:
            with self._driver.session() as session:
                records = session.run(cypher, **params)
                return [record.data() for record in records]
        except DriverError as e:
            logger.error(f"Cypher query failed: {e}")
            raise

    def clear(self) -> None:
        """Delete all nodes and relationships (use with care)."""
        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared all nodes and relationships")
        except DriverError as e:
            logger.error(f"Failed to clear graph: {e}")
            raise
