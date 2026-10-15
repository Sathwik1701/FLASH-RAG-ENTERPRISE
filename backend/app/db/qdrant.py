import logging
from typing import Optional
from qdrant_client import AsyncQdrantClient, models
from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantManager:
    _client: Optional[AsyncQdrantClient] = None

    @classmethod
    async def get_client(cls) -> AsyncQdrantClient:
        if cls._client is None:
            if settings.QDRANT_URL:
                try:
                    # Test whether the remote/docker instance is responsive
                    candidate_client = AsyncQdrantClient(
                        url=settings.QDRANT_URL,
                        api_key=settings.QDRANT_API_KEY or None,
                        timeout=2.0
                    )
                    await candidate_client.get_collections()
                    cls._client = candidate_client
                    logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
                except Exception as e:
                    logger.warning(
                        f"Failed to connect to Qdrant at {settings.QDRANT_URL} ({e}). "
                        f"Falling back to embedded local storage at '{settings.QDRANT_LOCAL_PATH}'."
                    )
                    cls._client = AsyncQdrantClient(path=settings.QDRANT_LOCAL_PATH)
            else:
                cls._client = AsyncQdrantClient(path=settings.QDRANT_LOCAL_PATH)
        return cls._client

    @classmethod
    async def init_collection(cls):
        """Initializes the collection with dense (HNSW) and sparse (Splade/BM25) vector indices."""
        client = await cls.get_client()
        collection_name = settings.QDRANT_COLLECTION_NAME

        collections_res = await client.get_collections()
        existing_names = [col.name for col in collections_res.collections]

        if collection_name not in existing_names:
            logger.info(f"Creating Qdrant collection: {collection_name}")
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=settings.effective_dense_dim,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False
                        )
                    )
                }
            )

            # Create payload indexes for fast filtering
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.document_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.file_hash",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Collection {collection_name} created successfully with hybrid indices.")
        else:
            logger.info(f"Collection {collection_name} already exists.")

    @classmethod
    async def check_hash_exists(cls, file_hash: str) -> Optional[str]:
        """Checks if a document with the given hash is already indexed. Returns document_id if found."""
        client = await cls.get_client()
        result, _ = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_hash",
                        match=models.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        if result and len(result) > 0:
            payload = result[0].payload
            return payload.get("metadata", {}).get("document_id")
        return None

    @classmethod
    async def close(cls):
        if cls._client is not None:
            await cls._client.close()
            cls._client = None
