from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import asyncio
import json
import hashlib
import os
from datetime import datetime
import structlog

from embeddings.embedding_service import EmbeddingService
from utils.cache import CacheManager
from utils.metrics import setup_metrics
from utils.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="Embedding Service",
    description="OpenAI-compatible embedding service with multiple model support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup metrics
metrics_registry = setup_metrics(app)

# Initialize services
embedding_service = EmbeddingService()
cache_manager = CacheManager()

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: str = Field(default="text-embedding-ada-002", description="Model to use")
    encoding_format: str = Field(default="float", regex="^(float|base64)$")
    user: Optional[str] = Field(default=None, description="User identifier")

class EmbeddingData(BaseModel):
    object: str = Field(default="embedding")
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = Field(default="list")
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for input text(s).
    Compatible with OpenAI embeddings API.
    """
    try:
        start_time = datetime.utcnow()
        
        # Normalize input to list
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        if not texts:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Check cache for existing embeddings
        cache_results = {}
        texts_to_embed = []
        text_indices = {}
        
        for i, text in enumerate(texts):
            cache_key = await cache_manager.generate_cache_key(text, request.model)
            cached_embedding = await cache_manager.get_embedding(cache_key)
            
            if cached_embedding is not None:
                cache_results[i] = cached_embedding
            else:
                texts_to_embed.append(text)
                text_indices[len(texts_to_embed) - 1] = i
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if texts_to_embed:
            new_embeddings = await embedding_service.create_embeddings(
                texts=texts_to_embed,
                model=request.model
            )
            
            # Cache new embeddings
            for j, embedding in enumerate(new_embeddings):
                original_index = text_indices[j]
                text = texts[original_index]
                cache_key = await cache_manager.generate_cache_key(text, request.model)
                await cache_manager.set_embedding(cache_key, embedding)
        
        # Combine cached and new embeddings
        all_embeddings = []
        new_embedding_index = 0
        
        for i in range(len(texts)):
            if i in cache_results:
                all_embeddings.append(cache_results[i])
            else:
                all_embeddings.append(new_embeddings[new_embedding_index])
                new_embedding_index += 1
        
        # Format response
        embedding_data = []
        for i, embedding in enumerate(all_embeddings):
            # Handle encoding format
            if request.encoding_format == "base64":
                import base64
                import struct
                embedding_bytes = b''.join(struct.pack('f', x) for x in embedding)
                embedding_encoded = base64.b64encode(embedding_bytes).decode('utf-8')
                embedding_data.append(EmbeddingData(
                    embedding=embedding_encoded,
                    index=i
                ))
            else:
                embedding_data.append(EmbeddingData(
                    embedding=embedding,
                    index=i
                ))
        
        # Calculate token usage
        total_tokens = sum(len(text.split()) for text in texts)
        
        response = EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Embeddings created",
            model=request.model,
            input_count=len(texts),
            cache_hits=len(cache_results),
            new_embeddings=len(new_embeddings),
            processing_time_ms=processing_time * 1000
        )
        
        return response
        
    except Exception as e:
        logger.error("Embedding creation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding creation failed: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available embedding models."""
    models = await embedding_service.list_available_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1699000000,
                "owned_by": info.get("owned_by", "rag-test-facade"),
                "permission": [],
                "root": model_id,
                "parent": None,
                **info
            }
            for model_id, info in models.items()
        ]
    }

@app.post("/v1/embeddings/batch")
async def create_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-ada-002",
    batch_size: int = 32
):
    """
    Create embeddings for large batches of text efficiently.
    """
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await embedding_service.create_embeddings(
                texts=batch,
                model=model
            )
            all_embeddings.extend(batch_embeddings)
        
        return {
            "embeddings": all_embeddings,
            "model": model,
            "count": len(all_embeddings)
        }
        
    except Exception as e:
        logger.error("Batch embedding creation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch embedding creation failed: {str(e)}")

@app.get("/v1/embeddings/similarity")
async def compute_similarity(
    text1: str,
    text2: str,
    model: str = "text-embedding-ada-002"
):
    """
    Compute semantic similarity between two texts.
    """
    try:
        embeddings = await embedding_service.create_embeddings(
            texts=[text1, text2],
            model=model
        )
        
        similarity = await embedding_service.compute_similarity(
            embeddings[0], embeddings[1]
        )
        
        return {
            "similarity": similarity,
            "model": model,
            "texts": [text1, text2]
        }
        
    except Exception as e:
        logger.error("Similarity computation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)