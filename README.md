# GraphRAG Compliance Engine

A Graph-based Retrieval-Augmented Generation engine for legal document analysis. It extracts compliance entities and relationships from legal text using an LLM, stores them in a Neo4j knowledge graph, and retrieves them via hybrid search combining vector similarity with graph traversal.

## Architecture

```
src/
├── ingestion/
│   └── extractor.py      # LLM-based entity & relationship extraction
├── storage/
│   └── graph.py           # Neo4j knowledge graph (KnowledgeGraph class)
└── retrieval/
    └── search.py          # Hybrid vector + graph search
tests/
└── test_retrieval.py      # 11 tests (mocked Neo4j, no live DB needed)
```

## Domain Model

### Nodes
| Type | Description | Example |
|------|-------------|---------|
| **Regulation** | A law, regulation, or standard | GDPR, SOX |
| **Company** | An organisation mentioned in text | Acme Corp |
| **Clause** | A specific article or section | Article 17, Section 302 |

### Edges
| Type | Description |
|------|-------------|
| **VIOLATES** | Source entity violates the target regulation/clause |
| **COMPLIES_WITH** | Source entity complies with the target regulation/clause |
| **REQUIRES** | A regulation/clause requires something of the target entity |

## Getting Started

### Prerequisites
- Python 3.10+
- Neo4j instance (local or remote) for production use
- An LLM API key (OpenAI, Anthropic, etc.) for entity extraction

### Installation

```bash
pip install -e ".[dev]"
```

### Running Tests

Tests are fully mocked and require no external services:

```bash
python -m pytest tests/ -v
```

## Usage

### 1. Extract entities from legal text

```python
from langchain_openai import ChatOpenAI
from src.ingestion.extractor import extract_entities_and_relationships

llm = ChatOpenAI(model="gpt-4o")
text = "Acme Corp was found to violate GDPR Article 17 regarding the right to erasure."

result = extract_entities_and_relationships(text, llm)
print(result.entities)       # [Entity(name='Acme Corp', type='Company'), ...]
print(result.relationships)  # [Relationship(source='Acme Corp', target='Article 17', type='VIOLATES')]
```

### 2. Store in Neo4j

```python
from src.storage.graph import KnowledgeGraph

kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
kg.add_extraction(result)
neighbours = kg.get_neighbours("GDPR")
kg.close()
```

### 3. Hybrid search

```python
from src.retrieval.search import VectorStore, hybrid_search

vector_store = VectorStore()
vector_store.add("Acme Corp was found to violate GDPR Article 17.")
vector_store.add("Article 5 of GDPR requires data minimisation.")

results = hybrid_search(
    query="What clauses violate GDPR?",
    vector_store=vector_store,
    knowledge_graph=kg,
)
print(results.vector_results)  # Top-k documents by similarity
print(results.graph_results)   # Neighbours from the knowledge graph
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` + `langchain-openai` | LLM orchestration |
| `neo4j` | Graph database driver |
| `pydantic` | Data validation (v2) |
| `pytest` | Testing (dev) |

## License

MIT
