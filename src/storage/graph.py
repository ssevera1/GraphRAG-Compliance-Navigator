"""Neo4j-backed knowledge graph for compliance triplets."""

from __future__ import annotations

import logging
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
                uri, auth=(user, password)
            )
            # Test connectivity on init to fail fast
            with self._driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j at {uri}: {e}")
            raise

    # ── public API ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()

    def add_entity(self, entity: Entity) -> None:
        """Merge a single entity node."""
        if not entity.name or not entity.name.strip():
            logger.warning(
                f"Skipping entity with empty name: {entity}"
            )
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
            logger.error(
                f"Failed to add entity '{entity.name}': {e}"
            )
            raise

    def add_relationship(self, rel: Relationship) -> None:
        """Merge a relationship between two already-existing nodes."""
        if not rel.source or not rel.target:
            logger.warning(
                f"Skipping relationship with empty source/target: {rel}"
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
                    query,
                    src=rel.source,
                    tgt=rel.target,
                    props=rel.properties,
                )
                # Check if nodes existed
                if result.consume().counters.relationships_created == 0:
                    logger.warning(
                        f"Relationship not created (nodes may not exist): "
                        f"{rel.source} -[{rel.type.value}]-> {rel.target}"
                    )
                else:
                    logger.debug(
                        f"Added relationship: {rel.source} -[{rel.type.value}]-> "
                        f"{rel.target}"
                    )
        except Exception as e:
            logger.error(
                f"Failed to add relationship {rel.source} -> {rel.target}: {e}"
            )
            raise

    def add_extraction(self, result: ExtractionResult) -> None:
        """Persist every entity and relationship from an extraction run."""
        if not result.entities and not result.relationships:
            logger.warning("Extraction result is empty")

        entity_count = len(result.entities)
        rel_count = len(result.relationships)

        for entity in result.entities:
            self.add_entity(entity)

        for rel in result.relationships:
            self.add_relationship(rel)

        logger.info(
            f"Added extraction: {entity_count} entities, {rel_count} relationships"
        )

    def get_neighbours(self, entity_name: str) -> list[dict]:
        """Return all directly connected nodes (any direction, any type)."""
        if not entity_name or not entity_name.strip():
            logger.warning(f"get_neighbours called with empty name")
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
            logger.debug(
                f"get_neighbours('{entity_name}'): found {len(results)} neighbours"
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to get neighbours for '{entity_name}': {e}"
            )
            return []

    def query(self, cypher: str, **params) -> list[dict]:
        """Run an arbitrary Cypher query and return results as dicts."""
        try:
            with self._driver.session() as session:
                records = session.run(cypher, **params)
                results = [record.data() for record in records]
            logger.debug(f"Cypher query returned {len(results)} records")
            return results
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            raise

    def clear(self) -> None:
        """Delete all nodes and relationships (use with care)."""
        try:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Graph cleared")
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            raise
