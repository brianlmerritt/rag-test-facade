import hashlib
import json
import pickle
from typing import List, Optional
import aioredis
import structlog
from utils.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

class CacheManager:
    """Redis-based cache manager for embeddings."""
    
    def __init__(self):
        self.redis_client = None
        self.cache_ttl = settings.cache_ttl
    
    async def _get_redis_client(self):
        """Get or create Redis client."""
        if self.redis_client is None:
            try:
                self.redis_client = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=False
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        return self.redis_client
    
    async def generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        # Create hash of text and model for consistent key
        content = f"{model}:{text}"
        cache_key = hashlib.sha256(content.encode()).hexdigest()
        return f"embedding:{cache_key}"
    
    async def get_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Retrieve embedding from cache."""
        try:
            client = await self._get_redis_client()
            if client is None:
                return None
            
            cached_data = await client.get(cache_key)
            if cached_data:
                # Deserialize embedding
                embedding = pickle.loads(cached_data)
                logger.debug(f"Cache hit for key: {cache_key}")
                return embedding
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def set_embedding(
        self, 
        cache_key: str, 
        embedding: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """Store embedding in cache."""
        try:
            client = await self._get_redis_client()
            if client is None:
                return False
            
            # Serialize embedding
            serialized_embedding = pickle.dumps(embedding)
            
            # Set with TTL
            ttl = ttl or self.cache_ttl
            await client.setex(cache_key, ttl, serialized_embedding)
            
            logger.debug(f"Cached embedding for key: {cache_key}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            client = await self._get_redis_client()
            if client is None:
                return 0
            
            keys = await client.keys(pattern)
            if keys:
                deleted_count = await client.delete(*keys)
                logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return 0
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await self._get_redis_client()
            if client is None:
                return {"status": "disconnected"}
            
            info = await client.info()
            
            # Get embedding-specific stats
            embedding_keys = await client.keys("embedding:*")
            
            return {
                "status": "connected",
                "total_keys": info.get("db0", {}).get("keys", 0),
                "embedding_keys": len(embedding_keys),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0)
            }
            
        except Exception as e:
            logger.warning(f"Cache stats retrieval failed: {e}")
            return {"status": "error", "error": str(e)}