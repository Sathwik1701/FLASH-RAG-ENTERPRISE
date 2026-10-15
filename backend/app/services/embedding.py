import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from openai import AsyncOpenAI
from fastembed import TextEmbedding
from app.core.config import settings
from app.services.sparse_encoder import fast_sparse_encoder

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    High-Throughput Dual Embedding Engine:
    - Dense: FastEmbed ONNX (BAAI/bge-small-en-v1.5) with multi-core parallelism
    - Sparse: FastBM25SparseEncoder (Sub-millisecond hashing, 0% OOM probability)
    """

    def __init__(self):
        self._openai_client: Optional[AsyncOpenAI] = None
        self._local_dense_model: Optional[TextEmbedding] = None

    @property
    def openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    @property
    def local_dense_model(self) -> TextEmbedding:
        if self._local_dense_model is None:
            logger.info(f"Loading Dense Embedding Model: {settings.LOCAL_EMBEDDING_MODEL}")
            self._local_dense_model = TextEmbedding(
                model_name=settings.LOCAL_EMBEDDING_MODEL,
                threads=4
            )
        return self._local_dense_model

    def _get_dense_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        dense_embs = self.local_dense_model.embed(texts, batch_size=batch_size)
        return [emb.tolist() for emb in dense_embs]

    async def get_dense_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            response = await self.openai_client.embeddings.create(
                input=texts,
                model=settings.DENSE_EMBEDDING_MODEL
            )
            return [item.embedding for item in response.data]
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_dense_batch, texts, batch_size)

    def get_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Ultra-fast BM25 sparse hashing (executes in milliseconds without blocking)."""
        return fast_sparse_encoder.encode(texts)

    async def get_sparse_embeddings_async(self, texts: List[str]) -> List[Dict[str, Any]]:
        return fast_sparse_encoder.encode(texts)

embedding_service = EmbeddingService()
