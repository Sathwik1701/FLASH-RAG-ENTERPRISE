import asyncio
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.models.retrieval import RetrievedChunk
from app.models.rerank import RankedChunk
from app.models.state import RAGWorkflowState
from app.services.reranker import reranker_service

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate chunks from hybrid retrieval using cross-attention,
    filters by confidence threshold, and bounds the context window.
    """

    @staticmethod
    async def rerank(
        query: str,
        candidate_chunks: List[RetrievedChunk],
        top_n: int = 5,
        score_threshold: float = 0.35
    ) -> List[RankedChunk]:
        # Edge Case 1: Empty input candidates
        if not candidate_chunks:
            logger.info("Candidate chunks list is empty. Skipping reranking.")
            return []

        passages = [chunk.text for chunk in candidate_chunks]

        # Execute CPU/GPU compute in threadpool to keep async event loop non-blocking
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(
                None,
                reranker_service.score_pairs,
                query,
                passages
            )
        except Exception as e:
            logger.error(f"Error executing cross-encoder inference: {e}", exc_info=True)
            # Degrade gracefully by using retrieval RRF scores
            scores = [chunk.score for chunk in candidate_chunks]

        ranked_list: List[RankedChunk] = []
        for idx, chunk in enumerate(candidate_chunks):
            relevance_score = scores[idx] if idx < len(scores) else 0.0

            # Threshold filtering: discard chunks below cutoff
            if relevance_score >= score_threshold:
                ranked_list.append(
                    RankedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        section_header=chunk.section_header,
                        has_table=chunk.has_table,
                        has_image=chunk.has_image,
                        relevance_score=relevance_score,
                        metadata=chunk.metadata
                    )
                )

        # Sort descending by relevance score
        ranked_list.sort(key=lambda x: x.relevance_score, reverse=True)

        # Truncate to top_n
        final_ranked = ranked_list[:top_n]

        # Edge Case 2: Insufficient context after threshold cutoff
        if not final_ranked:
            logger.warning(
                f"All {len(candidate_chunks)} candidate chunks fell below the threshold ({score_threshold}). "
                f"Top raw score was: {max(scores) if scores else 0.0:.4f}. Returning empty list to prevent hallucination."
            )

        logger.info(
            f"Reranked {len(candidate_chunks)} candidates -> Kept {len(final_ranked)} chunks "
            f"(threshold >= {score_threshold}, top_n = {top_n})"
        )

        return final_ranked


async def reranker_node(state: RAGWorkflowState) -> Dict[str, Any]:
    """
    LangGraph Node: Cross-Encoder Reranker.
    Consumes state.intent.standalone_query and state.retrieved_chunks.
    Populates state.reranked_chunks.
    """
    query = state.intent.standalone_query if state.intent else state.user_query
    raw_candidates = state.retrieved_chunks or []

    # Reconstruct RetrievedChunk instances from state dicts
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
