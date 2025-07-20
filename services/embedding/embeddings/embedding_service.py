from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer
import openai
import cohere
import httpx
from transformers import AutoTokenizer, AutoModel
import torch

logger = structlog.get_logger()

class EmbeddingService:
    """Service for creating embeddings using various models and providers."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients for external providers."""
        # OpenAI client
        if hasattr(openai, 'api_key') and openai.api_key:
            self.clients['openai'] = openai
        
        # Cohere client
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if cohere_api_key:
            self.clients['cohere'] = cohere.Client(cohere_api_key)
    
    async def create_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model: Model identifier
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Route to appropriate embedding method based on model
        if model.startswith("text-embedding-"):
            return await self._create_openai_embeddings(texts, model)
        elif model.startswith("embed-"):
            return await self._create_cohere_embeddings(texts, model)
        elif "sentence-transformers" in model or model in [
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"
        ]:
            return await self._create_sentence_transformer_embeddings(texts, model)
        elif "huggingface" in model or "/" in model:
            return await self._create_huggingface_embeddings(texts, model)
        else:
            # Default to sentence transformers
            return await self._create_sentence_transformer_embeddings(texts, model)
    
    async def _create_openai_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Create embeddings using OpenAI API."""
        if 'openai' not in self.clients:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = await asyncio.to_thread(
                self.clients['openai'].Embedding.create,
                input=texts,
                model=model
            )
            
            embeddings = []
            for item in response['data']:
                embeddings.append(item['embedding'])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            # Fallback to local model
            fallback_model = "all-MiniLM-L6-v2"
            logger.warning(f"Falling back to {fallback_model}")
            return await self._create_sentence_transformer_embeddings(texts, fallback_model)
    
    async def _create_cohere_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Create embeddings using Cohere API."""
        if 'cohere' not in self.clients:
            raise ValueError("Cohere API key not configured")
        
        try:
            response = await asyncio.to_thread(
                self.clients['cohere'].embed,
                texts=texts,
                model=model.replace("embed-", ""),
                input_type="search_document"
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            # Fallback to local model
            fallback_model = "all-MiniLM-L6-v2"
            logger.warning(f"Falling back to {fallback_model}")
            return await self._create_sentence_transformer_embeddings(texts, fallback_model)
    
    async def _create_sentence_transformer_embeddings(
        self, 
        texts: List[str], 
        model: str
    ) -> List[List[float]]:
        """Create embeddings using Sentence Transformers."""
        try:
            # Load model if not cached
            if model not in self.models:
                # Map common model names
                model_name = self._map_model_name(model)
                self.models[model] = SentenceTransformer(model_name)
            
            # Create embeddings
            embeddings = await asyncio.to_thread(
                self.models[model].encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Sentence Transformer embedding failed: {e}")
            raise ValueError(f"Failed to create embeddings with {model}: {str(e)}")
    
    async def _create_huggingface_embeddings(
        self, 
        texts: List[str], 
        model: str
    ) -> List[List[float]]:
        """Create embeddings using Hugging Face Transformers."""
        try:
            # Load model and tokenizer if not cached
            if model not in self.models:
                self.tokenizers[model] = AutoTokenizer.from_pretrained(model)
                self.models[model] = AutoModel.from_pretrained(model)
            
            tokenizer = self.tokenizers[model]
            model_obj = self.models[model]
            
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model_obj(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    embeddings.append(embedding.numpy().tolist())
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Hugging Face embedding failed: {e}")
            raise ValueError(f"Failed to create embeddings with {model}: {str(e)}")
    
    def _map_model_name(self, model: str) -> str:
        """Map common model names to actual Sentence Transformers model names."""
        model_mapping = {
            "text-embedding-ada-002": "all-MiniLM-L6-v2",
            "ada-002": "all-MiniLM-L6-v2",
            "small": "all-MiniLM-L6-v2",
            "base": "all-mpnet-base-v2",
            "large": "all-mpnet-base-v2",
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
            "qa": "multi-qa-MiniLM-L6-cos-v1",
            "code": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        return model_mapping.get(model, model)
    
    async def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            raise ValueError(f"Failed to compute similarity: {str(e)}")
    
    async def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available embedding models."""
        models = {
            # OpenAI models
            "text-embedding-ada-002": {
                "provider": "openai",
                "dimensions": 1536,
                "max_tokens": 8191,
                "owned_by": "openai"
            },
            "text-embedding-3-small": {
                "provider": "openai",
                "dimensions": 1536,
                "max_tokens": 8191,
                "owned_by": "openai"
            },
            "text-embedding-3-large": {
                "provider": "openai",
                "dimensions": 3072,
                "max_tokens": 8191,
                "owned_by": "openai"
            },
            
            # Cohere models
            "embed-english-v3.0": {
                "provider": "cohere",
                "dimensions": 1024,
                "max_tokens": 512,
                "owned_by": "cohere"
            },
            "embed-multilingual-v3.0": {
                "provider": "cohere",
                "dimensions": 1024,
                "max_tokens": 512,
                "owned_by": "cohere"
            },
            
            # Sentence Transformers models
            "all-MiniLM-L6-v2": {
                "provider": "sentence-transformers",
                "dimensions": 384,
                "max_tokens": 512,
                "owned_by": "rag-test-facade"
            },
            "all-mpnet-base-v2": {
                "provider": "sentence-transformers",
                "dimensions": 768,
                "max_tokens": 512,
                "owned_by": "rag-test-facade"
            },
            "multi-qa-MiniLM-L6-cos-v1": {
                "provider": "sentence-transformers",
                "dimensions": 384,
                "max_tokens": 512,
                "owned_by": "rag-test-facade"
            },
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "provider": "sentence-transformers",
                "dimensions": 384,
                "max_tokens": 512,
                "owned_by": "rag-test-facade"
            }
        }
        
        return models