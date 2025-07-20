from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8002
    workers: int = 4
    
    # Redis configuration
    redis_url: str = "redis://redis:6379"
    cache_ttl: int = 3600
    
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    cohere_api_key: Optional[str] = os.getenv("COHERE_API_KEY")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    # Model settings
    default_model: str = "all-MiniLM-L6-v2"
    max_batch_size: int = 32
    max_sequence_length: int = 512
    
    # Performance settings
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    model_cache_size: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings