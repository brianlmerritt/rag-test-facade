from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Service configuration
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 4
    
    # Redis configuration
    redis_url: str = "redis://redis:6379"
    cache_ttl: int = 3600
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    
    # Processing limits
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_chunks: int = 1000
    
    # Performance settings
    batch_size: int = 32
    max_tokens: int = 4096
    
    class Config:
        env_file = ".env"
        case_sensitive = False

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings