import asyncio
import logging
from typing import List, Optional, Dict, Any
from qdrant_client import models
from app.db.qdrant import QdrantManager
from app.core.config import settings
from app.services.embedding import embedding_service
from app.models.intent import RetrievalFilters, ContentTypeFilter
from app.models.retrieval import RetrievedChunk
from app.models.state import RAGWorkflowState

logger = logging.getLogger(__name__)

def build_qdrant_filter(filters: Optional[RetrievalFilters]) -> Optional[models.Filter]:
    """
    Translates Pydantic RetrievalFilters into strict Qdrant payload filters.
    Applies hard scoping before vector distance scoring to eliminate partition bleeding.
    """
    if not filters:
        return None

    must_conditions = []

    # Filter by specific document IDs
    if filters.document_ids and len(filters.document_ids) > 0:
        if len(filters.document_ids) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=filters.document_ids[0])
                )
            )
        else:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchAny(any=filters.document_ids)
                )
            )

    # Filter by content type if explicitly requested
    if filters.content_type == ContentTypeFilter.TABLE:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.has_table",
                match=models.MatchValue(value=True)
            )
        )
    elif filters.content_type == ContentTypeFilter.IMAGE:
        must_conditions.append(
            models.FieldCondition(
                key="metadata.has_image",
                match=models.MatchValue(value=True)
            )
        )

    if not must_conditions:
        return None

    return models.Filter(must=must_conditions)


class HybridRetriever:
    """
    Executes parallel Dense and Sparse vector retrieval against Qdrant,
    fusing candidate matches using Reciprocal Rank Fusion (RRF).
    """

    @staticmethod
    async def retrieve(
        query: str,
        retrieval_filters: Optional[RetrievalFilters] = None,
        top_k: int = 30
    ) -> List[RetrievedChunk]:
        client = await QdrantManager.get_client()
        qdrant_filter = build_qdrant_filter(retrieval_filters)

        # 1. Concurrently generate dense and sparse query embeddings
        dense_task = embedding_service.get_dense_embeddings([query])
        
        # FastEmbed is sync/cpu-bound: execute in threadpool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        sparse_task = loop.run_in_executor(None, embedding_service.get_sparse_embeddings, [query])

        dense_vecs = None
        sparse_vecs = None

        try:
            dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)
            
            if isinstance(dense_res, Exception):
                logger.error(f"Dense embedding failed: {dense_res}")
                raise dense_res
            else:
                dense_vecs = dense_res[0] if dense_res else None

            if isinstance(sparse_res, Exception):
                logger.warning(f"Sparse embedding failed ({sparse_res}). Degrading to dense-only retrieval.")
                sparse_vecs = None
            else:
                sparse_vecs = sparse_res[0] if sparse_res else None

        except Exception as e:
            logger.error(f"Embedding generation error in retriever: {e}")
            raise

        if not dense_vecs:
            logger.warning("Empty dense vector produced. Returning empty candidates.")
            return []

        # 2. Execute Hybrid Query with RRF (or fallback to Dense if sparse failed)
        try:
            if sparse_vecs and sparse_vecs.get("indices"):
                sparse_query_vector = models.SparseVector(
                    indices=sparse_vecs["indices"],
                    values=sparse_vecs["values"]
                )

                # Qdrant native Reciprocal Rank Fusion query using prefetch
                response = await client.query_points(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vecs,
                            using="dense",
                            filter=qdrant_filter,
                            limit=top_k
                        ),
                        models.Prefetch(
                            query=sparse_query_vector,
                            using="sparse",
                            filter=qdrant_filter,
                            limit=top_k
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                )
            else:
                # Degraded dense-only search
                logger.info("Executing dense-only search fallback.")
                response = await client.query_points(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    query=dense_vecs,
                    using="dense",
                    filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True
                )

            # 3. Format into strongly-typed RetrievedChunk objects
            retrieved_chunks: List[RetrievedChunk] = []
            for point in response.points:
                payload = point.payload or {}
                text = payload.get("text", "")
                meta = payload.get("metadata", {})

                chunk = RetrievedChunk(
                    chunk_id=str(point.id),
                    document_id=meta.get("document_id", "unknown"),
                    text=text,
                    page_number=meta.get("page_number", 1),
                    chunk_index=meta.get("chunk_index", 0),
                    section_header=meta.get("section_header"),
                    has_table=meta.get("has_table", False),
                    has_image=meta.get("has_image", False),
                    score=float(point.score) if point.score is not None else 0.0,
                    metadata=meta
                )
                retrieved_chunks.append(chunk)

            logger.info(f"Hybrid retrieval fetched {len(retrieved_chunks)} candidate chunks (Top-{top_k})")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}", exc_info=True)
            return []


async def hybrid_retrieval_node(state: RAGWorkflowState) -> Dict[str, Any]:
    """
    LangGraph Node: Hybrid Retrieval Execution.
    Consumes state.intent.standalone_query and state.intent.retrieval_filters.
    Propagates candidate points into state.retrieved_chunks.
    """
    if not state.intent:
        logger.warning("No intent found in state. Skipping retrieval.")
        return {"retrieved_chunks": []}

    standalone_query = state.intent.standalone_query
    filters = state.intent.retrieval_filters

    chunks = await HybridRetriever.retrieve(
        query=standalone_query,
        retrieval_filters=filters,
        top_k=30
    )

    # Convert to list of dicts for serializable LangGraph state
    return {"retrieved_chunks": [chunk.model_dump() for chunk in chunks]}
