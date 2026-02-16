"""Hybrid retrieval: vector similarity + graph traversal."""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from src.storage.graph import KnowledgeGraph


# ── Dummy embeddings (for tests — no API key required) ───────────────────────

class DummyEmbeddings(Embeddings):
    """Deterministic 64-dim mock embeddings derived from MD5 hashes.

    Useful for unit tests where semantic similarity is not required.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        return [int(c, 16) / 15.0 for c in digest[:64].ljust(64, "0")]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── In-memory vector store ──────────────────────────────────────────────────

def _default_embedder() -> Embeddings:
    """Lazy-load HuggingFaceEmbeddings so the import cost is paid only when needed."""
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: WPS433

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@dataclass
class VectorStore:
    """Minimal in-memory vector index.

    Parameters
    ----------
    embedder:
        Any LangChain ``Embeddings`` implementation.  Defaults to
        ``HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`` when
        *None* is provided.
    """

    embedder: Embeddings = field(default=None)  # type: ignore[assignment]
    documents: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = _default_embedder()

    def add(self, text: str) -> None:
        self.documents.append(text)
        self.embeddings.append(self.embedder.embed_query(text))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self.embedder.embed_query(query)
        scored = [
            {"text": doc, "score": cosine_similarity(q_emb, emb)}
            for doc, emb in zip(self.documents, self.embeddings)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ── Entity mention detection ────────────────────────────────────────────────

def _extract_entity_names_from_query(query: str) -> list[str]:
    """Heuristic: pull capitalised multi-word names and known acronyms."""
    # Match sequences of capitalised words (e.g. "Acme Corp") and
    # uppercase acronyms (e.g. "GDPR").
    tokens = re.findall(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b", query)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ── Hybrid search ────────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    vector_results: list[dict] = field(default_factory=list)
    graph_results: list[dict] = field(default_factory=list)


def hybrid_search(
    query: str,
    vector_store: VectorStore,
    knowledge_graph: KnowledgeGraph,
    top_k: int = 5,
) -> HybridResult:
    """Run vector similarity and graph traversal **in parallel**.

    Parameters
    ----------
    query:
        Natural-language question.
    vector_store:
        In-memory vector index to search against.
    knowledge_graph:
        Neo4j-backed graph for neighbourhood lookups.
    top_k:
        Number of vector results to return.

    Returns
    -------
    HybridResult
        Combined results from both retrieval paths.
    """
    entity_names = _extract_entity_names_from_query(query)

    def _vector_search() -> list[dict]:
        return vector_store.search(query, top_k=top_k)

    def _graph_search() -> list[dict]:
        results: list[dict] = []
        for name in entity_names:
            neighbours = knowledge_graph.get_neighbours(name)
            results.extend(neighbours)
        return results

    hybrid = HybridResult()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_vector_search): "vector",
            pool.submit(_graph_search): "graph",
        }
        for future in as_completed(futures):
            label = futures[future]
            if label == "vector":
                hybrid.vector_results = future.result()
            else:
                hybrid.graph_results = future.result()

    return hybrid
