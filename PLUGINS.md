# RAG Test Facade - Plugin Development Guide

This document explains how to create plugins for each service in the RAG Test Facade system. The facade pattern allows you to easily swap implementations while maintaining consistent APIs.

## 🏗️ System Architecture Overview

The RAG Test Facade system is composed of the following services:

| Service | Port | Purpose | Plugin Interface |
|---------|------|---------|------------------|
| API Gateway | 8000 | Request routing and load balancing | `GatewayInterface` |
| Document Processor | 8001 | Document parsing and chunking | `DocumentProcessorInterface` |
| Embedding Service | 8002 | Text-to-vector conversion | `EmbeddingInterface` |
| Sparse Retrieval | 8003 | Keyword/lexical search (BM25, SPLADE) | `SparseRetrieverInterface` |
| Dense Retrieval | 8004 | Vector similarity search | `DenseRetrieverInterface` |
| Graph Retrieval | 8005 | Knowledge graph traversal | `GraphRetrieverInterface` |
| Hybrid Search | 8006 | Multi-strategy search orchestration | `HybridSearchInterface` |
| Reranker | 8007 | Result refinement and scoring | `RerankerInterface` |
| Query Transform | 8008 | Query enhancement and expansion | `QueryTransformInterface` |
| RAG Generation | 8009 | Context assembly and response generation | `GenerationInterface` |
| Evaluation | 8010 | Performance metrics and monitoring | `EvaluatorInterface` |

## 🔌 Plugin Development Patterns

### Base Plugin Structure

All plugins should follow this structure:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()

class BaseInterface(ABC):
    """Base interface for all plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health status."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass
```

---

## 📄 Document Processor Plugins

### Interface Definition

```python
from typing import List, Dict, Any
from pydantic import BaseModel

class ChunkingStrategy(BaseModel):
    name: str
    chunk_size: int
    overlap: int
    metadata: Dict[str, Any]

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class DocumentProcessorInterface(BaseInterface):
    @abstractmethod
    async def parse_document(
        self, 
        content: bytes, 
        format: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Parse document content to text."""
        pass
    
    @abstractmethod
    async def create_chunks(
        self,
        text: str,
        strategy: ChunkingStrategy
    ) -> List[DocumentChunk]:
        """Create chunks from text."""
        pass
```

### Example Plugin: Custom PDF Parser

```python
class CustomPDFProcessor(DocumentProcessorInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.parser_config = config.get('pdf_parser', {})
    
    async def initialize(self) -> bool:
        # Initialize your PDF parsing library
        return True
    
    async def parse_document(self, content: bytes, format: str, metadata: Dict[str, Any]) -> str:
        if format != 'pdf':
            raise ValueError(f"Unsupported format: {format}")
        
        # Your custom PDF parsing logic
        text = your_pdf_parser(content, **self.parser_config)
        return text
    
    async def create_chunks(self, text: str, strategy: ChunkingStrategy) -> List[DocumentChunk]:
        # Your custom chunking logic
        chunks = your_chunking_algorithm(text, strategy)
        return chunks
```

### Registration

```python
# In your service main.py
from plugins.custom_pdf_processor import CustomPDFProcessor

# Register plugin
processor_registry = {
    'custom-pdf-v1': CustomPDFProcessor,
    'default': DefaultProcessor
}
```

---

## 🔢 Embedding Service Plugins

### Interface Definition

```python
class EmbeddingInterface(BaseInterface):
    @abstractmethod
    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        batch_size: int = 32
    ) -> List[List[float]]:
        """Create embeddings for texts."""
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self, model: str) -> int:
        """Get embedding dimension for model."""
        pass
    
    @abstractmethod
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List available models."""
        pass
```

### Example Plugin: Custom Embedding Provider

```python
class CustomEmbeddingProvider(EmbeddingInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.client = None
    
    async def initialize(self) -> bool:
        self.client = YourEmbeddingClient(
            api_key=self.api_key,
            base_url=self.base_url
        )
        return await self.client.test_connection()
    
    async def create_embeddings(
        self, 
        texts: List[str], 
        model: str, 
        batch_size: int = 32
    ) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.client.embed(batch, model)
            embeddings.extend(batch_embeddings)
        return embeddings
    
    async def get_embedding_dimension(self, model: str) -> int:
        return await self.client.get_model_info(model)['dimensions']
    
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        return await self.client.list_models()
```

---

## 🔍 Sparse Retrieval Plugins

### Interface Definition

```python
class SearchResult(BaseModel):
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    highlights: List[str] = []

class SparseRetrieverInterface(BaseInterface):
    @abstractmethod
    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Index documents for search."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from index."""
        pass
```

### Example Plugin: Custom Search Engine

```python
class CustomSearchEngine(SparseRetrieverInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index_name = config.get('index_name', 'documents')
        self.search_client = None
    
    async def initialize(self) -> bool:
        self.search_client = YourSearchEngine(
            host=self.config.get('host'),
            index=self.index_name
        )
        return await self.search_client.ping()
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        # Your indexing logic
        return await self.search_client.bulk_index(documents)
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        # Your search logic
        results = await self.search_client.search(
            query=query,
            size=top_k,
            filters=filters
        )
        
        return [
            SearchResult(
                document_id=result['id'],
                score=result['score'],
                text=result['text'],
                metadata=result['metadata'],
                highlights=result.get('highlights', [])
            )
            for result in results
        ]
```

---

## 🎯 Dense Retrieval Plugins

### Interface Definition

```python
class DenseRetrieverInterface(BaseInterface):
    @abstractmethod
    async def index_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        document_ids: List[str]
    ) -> bool:
        """Index embeddings."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search by embedding similarity."""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass
```

### Example Plugin: Custom Vector Database

```python
class CustomVectorDB(DenseRetrieverInterface):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.collection_name = config.get('collection_name', 'embeddings')
        self.vector_client = None
    
    async def initialize(self) -> bool:
        self.vector_client = YourVectorDB(
            host=self.config.get('host'),
            port=self.config.get('port')
        )
        return await self.vector_client.connect()
    
    async def index_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        document_ids: List[str]
    ) -> bool:
        # Your vector indexing logic
        return await self.vector_client.insert(
            collection=self.collection_name,
            vectors=embeddings,
            metadata=metadata,
            ids=document_ids
        )
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        # Your vector search logic
        results = await self.vector_client.search(
            collection=self.collection_name,
            vector=query_embedding,
            limit=top_k,
            filters=filters
        )
        
        return [
            SearchResult(
                document_id=result['id'],
                score=result['distance'],
                text=result['metadata']['text'],
                metadata=result['metadata']
            )
            for result in results
        ]
```

---

## 📊 Graph Retrieval Plugins

### Interface Definition

```python
class GraphNode(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any]
    relevance_score: float

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class GraphResult(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    natural_language_context: str

class GraphRetrieverInterface(BaseInterface):
    @abstractmethod
    async def query_graph(
        self,
        query: str,
        max_hops: int = 2,
        entity_types: List[str] = None,
        relationship_types: List[str] = None
    ) -> GraphResult:
        """Query knowledge graph."""
        pass
    
    @abstractmethod
    async def add_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> bool:
        """Add entities to graph."""
        pass
    
    @abstractmethod
    async def add_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> bool:
        """Add relationships to graph."""
        pass
```

---

## 🔄 Hybrid Search Plugins

### Interface Definition

```python
class HybridSearchStrategy(BaseModel):
    sparse_weight: float
    dense_weight: float
    graph_weight: float
    fusion_method: str  # 'weighted', 'rrf', 'dynamic'

class HybridSearchInterface(BaseInterface):
    @abstractmethod
    async def search(
        self,
        query: str,
        strategy: HybridSearchStrategy,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Perform hybrid search."""
        pass
    
    @abstractmethod
    async def optimize_weights(
        self,
        queries: List[str],
        relevance_scores: List[List[float]]
    ) -> HybridSearchStrategy:
        """Optimize fusion weights."""
        pass
```

---

## 🏆 Reranker Plugins

### Interface Definition

```python
class RerankResult(BaseModel):
    document_id: str
    relevance_score: float
    original_rank: int
    reranked_rank: int

class RerankerInterface(BaseInterface):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[SearchResult],
        top_k: int = 5
    ) -> List[RerankResult]:
        """Rerank search results."""
        pass
    
    @abstractmethod
    async def batch_rerank(
        self,
        queries: List[str],
        document_batches: List[List[SearchResult]]
    ) -> List[List[RerankResult]]:
        """Batch rerank multiple queries."""
        pass
```

---

## 🔄 Query Transform Plugins

### Interface Definition

```python
class QueryTransformation(BaseModel):
    strategy: str
    queries: List[str] = []
    hypothetical_document: str = ""
    reasoning: str = ""

class QueryTransformInterface(BaseInterface):
    @abstractmethod
    async def transform_query(
        self,
        query: str,
        strategies: List[str]
    ) -> List[QueryTransformation]:
        """Transform query using multiple strategies."""
        pass
    
    @abstractmethod
    async def classify_query(
        self,
        query: str
    ) -> Dict[str, Any]:
        """Classify query type and complexity."""
        pass
```

---

## 💬 Generation Service Plugins

### Interface Definition

```python
class GenerationInterface(BaseInterface):
    @abstractmethod
    async def generate_response(
        self,
        query: str,
        context: List[SearchResult],
        model: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate response with RAG context."""
        pass
    
    @abstractmethod
    async def stream_response(
        self,
        query: str,
        context: List[SearchResult],
        model: str
    ) -> AsyncGenerator[str, None]:
        """Stream response generation."""
        pass
```

---

## 📈 Evaluation Plugins

### Interface Definition

```python
class EvaluationMetric(BaseModel):
    name: str
    value: float
    description: str

class EvaluatorInterface(BaseInterface):
    @abstractmethod
    async def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[SearchResult]],
        relevant_docs: List[List[str]]
    ) -> List[EvaluationMetric]:
        """Evaluate retrieval performance."""
        pass
    
    @abstractmethod
    async def evaluate_generation(
        self,
        queries: List[str],
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> List[EvaluationMetric]:
        """Evaluate generation quality."""
        pass
```

---

## 🔧 Plugin Registration and Discovery

### Plugin Registry Pattern

```python
# In each service's main.py
class PluginRegistry:
    def __init__(self):
        self.plugins = {}
    
    def register(self, name: str, plugin_class):
        self.plugins[name] = plugin_class
    
    def get(self, name: str, config: Dict[str, Any]):
        if name not in self.plugins:
            raise ValueError(f"Plugin {name} not found")
        return self.plugins[name](config)
    
    def list_plugins(self) -> List[str]:
        return list(self.plugins.keys())

# Usage
registry = PluginRegistry()
registry.register('custom-embedder', CustomEmbeddingProvider)
registry.register('default-embedder', DefaultEmbeddingProvider)

# Load plugin based on config
embedder = registry.get(config['embedding_provider'], config)
```

### Environment-Based Plugin Selection

```yaml
# In docker-compose.yml
services:
  embedding-service:
    environment:
      - EMBEDDING_PROVIDER=custom-embedder
      - CUSTOM_EMBEDDER_API_KEY=${CUSTOM_API_KEY}
      - CUSTOM_EMBEDDER_BASE_URL=${CUSTOM_BASE_URL}
```

### Plugin Configuration

```yaml
# plugins.yaml
plugins:
  embedding_service:
    provider: custom-embedder
    config:
      api_key: ${CUSTOM_API_KEY}
      base_url: ${CUSTOM_BASE_URL}
      batch_size: 64
      timeout: 30
  
  sparse_retrieval:
    provider: custom-search
    config:
      host: ${SEARCH_HOST}
      index_name: documents_v2
      max_results: 1000
```

## 🧪 Testing Plugins

### Plugin Test Template

```python
import pytest
from your_plugin import YourCustomPlugin

@pytest.fixture
async def plugin():
    config = {
        'test_config': 'value'
    }
    plugin = YourCustomPlugin(config)
    await plugin.initialize()
    yield plugin
    await plugin.cleanup()

class TestYourCustomPlugin:
    async def test_initialize(self, plugin):
        assert await plugin.health_check()
    
    async def test_core_functionality(self, plugin):
        # Test your plugin's core functionality
        result = await plugin.your_method('test_input')
        assert result is not None
    
    async def test_error_handling(self, plugin):
        # Test error scenarios
        with pytest.raises(ValueError):
            await plugin.your_method('invalid_input')
```

## 📦 Plugin Packaging

### Directory Structure
```
plugins/
├── your_plugin/
│   ├── __init__.py
│   ├── plugin.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── README.md
│   └── tests/
│       └── test_plugin.py
```

### Plugin Metadata

```python
# In your plugin's __init__.py
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Custom plugin for XYZ functionality"
__requires__ = ["service_version>=1.0.0"]

from .plugin import YourCustomPlugin

__all__ = ['YourCustomPlugin']
```

## 🚀 Best Practices

1. **Error Handling**: Always implement comprehensive error handling and logging
2. **Configuration**: Use typed configuration classes with validation
3. **Testing**: Write comprehensive unit and integration tests
4. **Documentation**: Document your plugin's API and configuration options
5. **Versioning**: Use semantic versioning for your plugins
6. **Performance**: Consider async/await patterns and batch processing
7. **Security**: Validate all inputs and implement proper authentication
8. **Monitoring**: Add metrics and health checks to your plugins

## 📚 Resources

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://docs.pydantic.dev/)
- [AsyncIO Best Practices](https://docs.python.org/3/library/asyncio.html)
- [Docker Plugin Development](https://docs.docker.com/engine/extend/)

---

For more examples and advanced patterns, see the `examples/plugins/` directory in this repository.