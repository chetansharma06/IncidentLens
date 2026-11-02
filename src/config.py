from functools import lru_cache
from pathlib import Path
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings(BaseModel):
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))
    embedding_dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "3072")))

    # Pinecone
    pinecone_api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index_name: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "cloudops-sentinel-gemini-self-rag"))
    pinecone_namespace: str = Field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "incident-runbooks"))
    pinecone_cloud: str = Field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = Field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))

    # Internet search
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # Self-RAG controls
    top_k: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    max_support_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_SUPPORT_RETRIES", "2")))
    max_retrieval_rewrites: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIEVAL_REWRITES", "2")))
    max_web_rewrites: int = Field(default_factory=lambda: int(os.getenv("MAX_WEB_REWRITES", "2")))
    database_path: str = Field(default_factory=lambda: os.getenv("DATABASE_PATH", "data/audit.db"))

    @property
    def database_file(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else ROOT / p



@lru_cache
def get_settings() -> Settings:
    return Settings()
