from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API Server
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: List[str] = ["*"]

    # LLM Provider: "groq" (primary), "nvidia", or "openai"
    LLM_PROVIDER: str = "groq"

    # Groq Configuration (Ultra-fast inference)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_TIMEOUT: float = 10.0

    # OpenAI Keys & Models
    OPENAI_API_KEY: str = ""
    ROUTER_MODEL: str = "gpt-4o-mini"
    GENERATION_MODEL: str = "gpt-4o"

    # Dense Embedding Provider: "huggingface" or "openai"
    EMBEDDING_PROVIDER: str = "huggingface"
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    LOCAL_EMBEDDING_DIM: int = 384

    @property
    def effective_dense_dim(self) -> int:
        return self.LOCAL_EMBEDDING_DIM

    # Qdrant Vector DB
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "rag_multimodal_documents"
    QDRANT_LOCAL_PATH: str = "./qdrant_local_data"

    # FlashRank Reranker Configuration
    RERANKER_MODEL_NAME: str = "ms-marco-TinyBERT-L-2-v2"
    RERANKER_SCORE_THRESHOLD: float = 0.25
    RERANKER_TOP_K: int = 6

settings = Settings()
