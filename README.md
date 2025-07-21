# RAG Test Facade System

A prototype, modular Retrieval-Augmented Generation (RAG) system built with the facade pattern. Provides OpenAI-compatible APIs while abstracting away implementation details for maximum flexibility.

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Plugin Development](#-plugin-development)
- [Monitoring](#-monitoring)
- [Development](#-development)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🏗️ Architecture

This system implements a microservices architecture with the following components:

### Core Services
- **API Gateway** (Port 8000) - Routes requests and provides unified API interface
- **Document Processor** (Port 8001) - Handles document ingestion and chunking
- **Embedding Service** (Port 8002) - Multi-model embedding generation
- **Sparse Retrieval** (Port 8003) - BM25 and lexical search
- **Dense Retrieval** (Port 8004) - Vector similarity search
- **Graph Retrieval** (Port 8005) - Knowledge graph traversal
- **Hybrid Search** (Port 8006) - Combines multiple retrieval strategies
- **Reranker** (Port 8007) - Result refinement and scoring
- **Query Transform** (Port 8008) - Query enhancement and expansion
- **RAG Generation** (Port 8009) - Context assembly and response generation
- **Evaluation** (Port 8010) - Performance metrics and monitoring

### Data Stores
- **Elasticsearch** (Port 9200) - Document indexing for sparse retrieval
- **Milvus** (Port 19530) - Vector database for dense retrieval
- **Neo4j** (Ports 7474/7687) - Graph database for knowledge graphs
- **Redis** (Port 6379) - Caching and session management
- **PostgreSQL** (Port 5432) - Evaluation data and metrics

### Monitoring
- **Prometheus** (Port 9090) - Metrics collection
- **Grafana** (Port 3000) - Dashboards and visualization

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- At least 8GB RAM (16GB+ recommended)
- NVIDIA GPU support (optional, for better performance)

### 1. Clone and Setup
```bash
git clone <repository>
cd rag-test-facade
```

### 2. Configure Environment
Copy and edit the environment file:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Required API keys (add to .env):
```bash
OPENAI_API_KEY=sk-your-openai-key
COHERE_API_KEY=your-cohere-key
HUGGINGFACE_API_KEY=your-hf-key
```

### 3. Start the System
```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

### 4. Verify Installation
```bash
# Test API Gateway
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

## 📖 Usage Examples

### Document Processing
```python
import requests

# Process a document
response = requests.post("http://localhost:8000/v1/documents/process", json={
    "input": "Your document content here",
    "format": "txt",
    "model": "chunker-v1",
    "options": {
        "chunk_size": 512,
        "overlap": 50,
        "strategy": "semantic"
    }
})

chunks = response.json()
```

### Create Embeddings
```python
# OpenAI-compatible embeddings
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": ["text to embed", "another text"],
    "model": "text-embedding-ada-002"
})

embeddings = response.json()
```

### Hybrid Search
```python
# Search across multiple strategies
response = requests.post("http://localhost:8000/v1/search/hybrid", json={
    "query": "What is machine learning?",
    "strategies": {
        "sparse": {"enabled": True, "weight": 0.3},
        "dense": {"enabled": True, "weight": 0.5},
        "graph": {"enabled": True, "weight": 0.2}
    },
    "fusion_method": "rrf",
    "top_k": 10
})

results = response.json()
```

### RAG Generation
```python
# OpenAI-compatible chat with RAG
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "rag_config": {
        "enabled": True,
        "search_strategy": "hybrid",
        "top_k": 5,
        "include_citations": True
    }
})

answer = response.json()
```

## 🔧 Configuration

### Chunking Strategies
- **semantic**: Groups related content using embeddings
- **fixed**: Fixed-size chunks with token counting
- **sentence**: Sentence-aware chunking
- **paragraph**: Paragraph-based splitting

### Embedding Models
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small/large
- **Cohere**: embed-english-v3.0, embed-multilingual-v3.0
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2
- **Custom**: Support for any HuggingFace model

### Search Fusion Methods
- **weighted**: Static weighted combination
- **rrf**: Reciprocal Rank Fusion
- **dynamic**: LLM-based adaptive weighting

## 📊 Monitoring

### Metrics Dashboard
Access Grafana at http://localhost:3000 (admin/admin)

### Health Checks
```bash
# Check all service health
curl http://localhost:8000/health

# Individual service health
curl http://localhost:8001/health  # Document processor
curl http://localhost:8002/health  # Embeddings
# ... etc
```

### Prometheus Metrics
Access raw metrics at http://localhost:9090

## 🛠️ Development

### Running Individual Services
```bash
# Start only core services
docker-compose up api-gateway document-processor embedding-service

# Start with specific data stores
docker-compose up elasticsearch milvus redis
```

### Adding New Models
1. Implement the model interface in the appropriate service
2. Update the model registry
3. Add configuration options
4. Test with the evaluation framework

### Custom Retrievers
Implement the retriever interface:
```python
class CustomRetriever(RetrieverInterface):
    async def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Your implementation
        pass
```

## 🔌 Plugin Development

The RAG Test Facade system is designed with extensibility in mind. You can create custom plugins for each service while maintaining OpenAI-compatible APIs.

**📖 [Complete Plugin Development Guide](PLUGINS.md)**

The plugin system supports:
- **Custom Document Processors** - Add support for new file formats and chunking strategies
- **Custom Embedding Providers** - Integrate any embedding model or API
- **Custom Retrievers** - Implement specialized search algorithms (sparse, dense, graph)
- **Custom Rerankers** - Add sophisticated result refinement models
- **Custom Generators** - Integrate different LLMs and generation strategies

Each plugin follows a consistent interface pattern, making it easy to swap implementations without changing client code.

## 🧪 Testing

### Run Evaluation Suite
```bash
# Evaluate retrieval performance
curl -X POST http://localhost:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_type": "retrieval",
    "test_cases": [...],
    "metrics": ["precision", "recall", "mrr"]
  }'
```

### Load Testing
```bash
# Install artillery
npm install -g artillery

# Run load tests
artillery run load-test.yml
```

## 🚨 Troubleshooting

### Common Issues

**Service won't start**
```bash
# Check logs
docker-compose logs <service-name>

# Restart specific service
docker-compose restart <service-name>
```

**Out of memory**
```bash
# Reduce batch sizes in .env
BATCH_SIZE=16
MAX_WORKERS=2

# Or increase Docker memory limit
```

**GPU not detected**
```bash
# Verify GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check NVIDIA Docker runtime
docker info | grep nvidia
```

### Performance Tuning

**Memory optimization**
- Reduce model cache size
- Use smaller embedding models
- Limit concurrent requests

**Speed optimization**
- Enable GPU acceleration
- Increase batch sizes
- Use faster embedding models
- Enable caching

## 📚 API Reference

Complete API documentation available at:
- Swagger UI: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows you to:
- ✅ Use the software for any purpose (commercial or non-commercial)
- ✅ Modify and distribute the software
- ✅ Include in proprietary software
- ✅ Sell copies of the software

**Attribution**: Please include the original copyright notice in any substantial portions of the software you distribute.

## 🆘 Support

- GitHub Issues: [Create an issue](link-to-issues)
- Documentation: [Full docs](link-to-docs)
- Discord: [Join our community](link-to-discord)

---

Built with ❤️ using Docker, FastAPI, and modern ML frameworks.
