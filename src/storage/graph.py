"""Neo4j-backed knowledge graph for compliance triplets."""

from __future__ import annotations

import time
from neo4j import GraphDatabase, Driver

from src.ingestion.extractor import Entity, ExtractionResult, Relationship


class KnowledgeGraph:
    """Thin wrapper around a Neo4j instance for storing compliance triplets.

    Usage::

        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
        kg.add_extraction(result)
        neighbours = kg.get_neighbours("GDPR")
        kg.close()
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: Driver | None = None
        max_retries = 5
        base_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                self._driver = GraphDatabase.driver(uri, auth=(user, password))
                # Validate connection by running a simple query
                with self._driver.session() as session:
                    session.run("RETURN 1")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect to Neo4j at {uri} after {max_retries} attempts"
                    ) from e

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
