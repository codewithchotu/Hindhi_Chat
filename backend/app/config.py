"""Application configuration management using Pydantic Settings."""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """System-wide configuration settings loaded from environment or .env file."""
    
    # App
    APP_NAME: str = "Cross-Lingual Hindi RAG Engine"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
    
    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_SOURCE_DIR: Path = BASE_DIR / "data" / "source"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "storage" / "chroma_db")
    COLLECTION_NAME: str = "hindi_rag_kalam"
    
    # Model Cache Directory (D: drive to protect C: disk limits)
    HF_HOME: str = os.getenv("HF_HOME", "D:/hf_cache")
    TORCH_HOME: str = os.getenv("TORCH_HOME", "D:/torch_cache")
    
    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    
    # Gemini LLM
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    # Chunking Defaults
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80
    
    # Retrieval Defaults
    TOP_K: int = 5
    HYBRID_ALPHA: float = 0.7  # 0.7 dense + 0.3 BM25
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


_settings = None


def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
