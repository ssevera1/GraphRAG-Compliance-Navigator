"""Hybrid retrieval: vector similarity + graph traversal."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from langchain_core.embeddings import Embeddings

from src.storage.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


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
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b:
        Embedding vectors.

    Returns
    -------
    float
        Similarity score in [0, 1], or 0.0 if either vector is None or empty.

    Raises
    ------
    ValueError
        If vectors have different lengths.
    """
    if a is None or b is None:
        return 0.0
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    
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
        if not self.documents:
            return []
        q_emb = self.embedder.embed_query(query)
        scored: list[dict[str, Any]] = [
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


# ── Retry logic ─────────────────────────────────────────────────────────────

T = TypeVar("T")


def _retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
) -> T:
    """Execute function with exponential backoff retry on failure.

    Parameters
    ----------
    func:
        Callable to execute.
    max_retries:
        Maximum number of retry attempts.
    initial_delay:
        Initial delay in seconds before first retry.
    backoff_factor:
        Multiplier for delay after each retry.

    Returns
    -------
    T
        Result of successful function execution.

    Raises
    ------
    Exception
        If all retries are exhausted.
    """
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= backoff_factor
            else:
                break

    raise last_exception if last_exception else RuntimeError(
        "Function failed after retries with no exception captured"
    )


# ── Hybrid search ────────────────────────────────────────────────────────────

def _is_valid_neighbour(neighbour: Any) -> bool:
    """Check if a neighbour structure is well-formed.
    
    Parameters
    ----------
    neighbour:
        A neighbour result from the knowledge graph.
    
    Returns
    -------
    bool
        True if the neighbour is a non-empty dict with required structure.
    """
    if not isinstance(neighbour, dict):
        return False
    if not neighbour:
        return False
    return True


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
        return _retry_with_backoff(lambda: vector_store.search(query, top_k=top_k))

    def _graph_search() -> list[dict]:
        if not entity_names:
            return []

        def _search_with_retry() -> list[dict]:
            results: list[dict] = []
            for name in entity_names:
                neighbours = knowledge_graph.get_neighbours(name)

                if neighbours is None:
                    logger.warning(
                        "get_neighbours(%r) returned None, expected an iterable "
                        "of dicts; skipping this entity",
                        name,
                    )
                    continue

                # Deliberately an iterable check rather than isinstance(list):
                # get_neighbours currently materialises a list comprehension
                # over the Neo4j cursor, but streaming those records instead is
                # a plausible refactor, and a stricter check would then discard
                # every neighbour for every entity.
                if isinstance(neighbours, (str, bytes)) or not isinstance(
                    neighbours, Iterable
                ):
                    logger.warning(
                        "get_neighbours(%r) returned %s, expected an iterable "
                        "of dicts; skipping this entity",
                        name,
                        type(neighbours).__name__,
                    )
                    continue

                for neighbour in neighbours:
                    if not _is_valid_neighbour(neighbour):
                        logger.warning(
                            "get_neighbours(%r) yielded an invalid record (%s); "
                            "skipping that record",
                            name,
                            "empty dict"
                            if isinstance(neighbour, dict)
                            else type(neighbour).__name__,
                        )
                        continue
                    results.append(neighbour)
            return results

        return _retry_with_backoff(_search_with_retry)

    hybrid = HybridResult()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_vector_search): "vector",
            pool.submit(_graph_search): "graph",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result(timeout=30)
                if label == "vector":
                    hybrid.vector_results = result
                else:
                    hybrid.graph_results = result
            except Exception:
                if label == "vector":
                    hybrid.vector_results = []
                else:
                    hybrid.graph_results = []

    return hybrid
