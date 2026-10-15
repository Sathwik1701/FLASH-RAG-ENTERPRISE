import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.generation import QueryStreamRequest
from app.rag.graph import execute_rag_pipeline_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])

@router.post("/stream")
async def stream_query_response(request: QueryStreamRequest):
    """
    Real-time SSE Endpoint:
    Orchestrates the query through LangGraph (Intent -> Hybrid Retrieval -> Reranking)
    and streams back tokens and citation badges via Server-Sent Events.
    """
    logger.info(f"Incoming SSE query request: '{request.query}' with {len(request.selected_documents)} docs selected.")

    generator = execute_rag_pipeline_stream(
        user_query=request.query,
        chat_history=request.chat_history,
        selected_documents=request.selected_documents
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering in Nginx / Cloudflare
            "Content-Type": "text/event-stream; charset=utf-8"
        }
    )
