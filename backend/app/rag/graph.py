import logging
import json
import time
from typing import AsyncGenerator
from app.rag.intent_node import evaluate_fast_intent
from app.rag.retrieval_node import HybridRetriever
from app.rag.reranker_node import CrossEncoderReranker
from app.rag.generator_node import generate_response_stream
from app.models.intent import RouteDecision
from app.core.config import settings

logger = logging.getLogger(__name__)

async def execute_rag_pipeline_stream(
    user_query: str,
    chat_history: list,
    selected_documents: list
) -> AsyncGenerator[str, None]:
    """
    Sub-second Optimized RAG Execution Pipeline:
    1. Status Telemetry: {"type": "status", "stage": "searching"} (< 10ms)
    2. Dual-Track Optimistic Routing: Intent resolved via deterministic rules (< 1ms).
    3. Parallel Qdrant Hybrid Retrieval + FlashRank Reranker (max 4 chunks).
    4. Status Telemetry: {"type": "status", "stage": "retrieved", "count": N} (~300-800ms)
    5. Direct Groq Token Streaming (TTFT < 500ms).
    """
    t_start = time.perf_counter()

    # Telemetry: Immediate acknowledgement
    yield f"data: {json.dumps({'type': 'status', 'stage': 'searching'})}\n\n"

    # Step 1: Fast deterministic intent evaluation (< 1ms)
    intent = evaluate_fast_intent(
        query=user_query,
        chat_history=chat_history,
        selected_docs=selected_documents
    )

    route = intent.route

    # If conversational or rejected, skip vector retrieval entirely
    if route != RouteDecision.RAG_QUERY:
        async for chunk in generate_response_stream(
            user_query=user_query,
            route=route,
            reranked_chunks=[],
            chat_history=chat_history
        ):
            yield chunk
        return

    # Step 2: High-speed Hybrid Retrieval
    t_ret = time.perf_counter()
    raw_candidates = await HybridRetriever.retrieve(
        query=intent.standalone_query,
        retrieval_filters=intent.retrieval_filters,
        top_k=20
    )
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    # Step 3: FlashRank Micro-Reranker (Top 3-4 chunks)
    t_rerank = time.perf_counter()
    ranked_chunks = await CrossEncoderReranker.rerank(
        query=intent.standalone_query,
        candidate_chunks=raw_candidates,
        top_n=settings.RERANKER_TOP_K,
        score_threshold=settings.RERANKER_SCORE_THRESHOLD
    )
    rerank_ms = (time.perf_counter() - t_rerank) * 1000

    # Telemetry: Retrieved chunks count
    yield f"data: {json.dumps({'type': 'status', 'stage': 'retrieved', 'count': len(ranked_chunks), 'retrieval_ms': round(retrieval_ms, 1), 'rerank_ms': round(rerank_ms, 1)})}\n\n"

    # Step 4: Stream Tokens via Groq
    chunk_dicts = [c.model_dump() for c in ranked_chunks]

    async for chunk in generate_response_stream(
        user_query=user_query,
        route=route,
        reranked_chunks=chunk_dicts,
        chat_history=chat_history
    ):
        yield chunk
