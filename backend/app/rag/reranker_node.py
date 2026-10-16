import logging
from typing import List, Dict, Any
from app.core.config import settings
from app.models.retrieval import RetrievedChunk
from app.models.rerank import RankedChunk
from app.models.state import RAGWorkflowState
from app.services.rrf_reranker import algorithmic_reranker

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate chunks using Algorithmic Reciprocal Rank Fusion & Lexical Density.
    Consumes 0 MB of extra memory, preventing Render 512 MB OOM crashes.
    """

    @staticmethod
    async def rerank(
        query: str,
        candidate_chunks: List[RetrievedChunk],
        top_n: int = 6,
        score_threshold: float = 0.20
    ) -> List[RankedChunk]:
        return algorithmic_reranker.rerank(
            query=query,
            candidate_chunks=candidate_chunks,
            top_n=top_n,
            score_threshold=score_threshold
        )


async def reranker_node(state: RAGWorkflowState) -> Dict[str, Any]:
    """
    LangGraph Node: Algorithmic RRF Reranker.
    Consumes state.intent.standalone_query and state.retrieved_chunks.
    Populates state.reranked_chunks.
    """
    query = state.intent.standalone_query if state.intent else state.user_query
    raw_candidates = state.retrieved_chunks or []

    candidate_objects = [RetrievedChunk(**chunk_dict) for chunk_dict in raw_candidates]

    top_n = settings.RERANKER_TOP_K
    threshold = settings.RERANKER_SCORE_THRESHOLD

    ranked_results = await CrossEncoderReranker.rerank(
        query=query,
        candidate_chunks=candidate_objects,
        top_n=top_n,
        score_threshold=threshold
    )

    return {"reranked_chunks": [item.model_dump() for item in ranked_results]}

