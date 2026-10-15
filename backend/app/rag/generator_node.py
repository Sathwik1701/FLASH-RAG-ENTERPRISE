import json
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.rerank import RankedChunk
from app.models.generation import CitationBadge
from app.models.intent import RouteDecision

logger = logging.getLogger(__name__)

# Static system prompt positioned first for provider prompt-caching
SYSTEM_PROMPT = """You are an expert Enterprise Document Analyst and Grounded Attribution Engine.
Your mission is to provide thorough, structured, and comprehensive answers synthesized from the provided document context.

### GUIDELINES FOR ACCURACY & SYNTHESIS:
1. FACTUAL GROUNDING: Rely strictly on the facts, specifications, tables, and discussions in the context blocks.
2. SYNTHESIS & REASONING: When asked comparative or evaluative questions (e.g. "which model has better architecture" or "which is more efficient"), analyze the documented specifications and metrics. Present the comparison clearly with structured sections, pros/cons, and tradeoffs based on what the document describes.
3. CITATIONS: Attribute key technical claims and metrics using the exact inline citation format: `[Doc: <Filename>, Page: <PageNumber>]` using the filename from the context block header.
4. HONEST UNCERTAINTY: Only if the context is completely irrelevant to the question, explain that the current documents do not mention the topic. Otherwise, synthesize all available relevant data.
5. FORMATTING & READABILITY: 
   - Structure answers into numbered sections (`### 1. Section Title`).
   - Use bold lead-ins for key points.
   - Use Markdown tables (`| Col1 | Col2 |`) when comparing metrics across models or entities.
   - Avoid raw, unparsed LaTeX strings; write mathematical expressions simply and legibly (e.g. `RMSNorm(x) = x / RMS(x) * gamma`).
"""

CONVERSATIONAL_PROMPT = """You are an intelligent, friendly enterprise RAG assistant.
- If the user greets you, greet them warmly.
- If the user asks what you can do, who you are, or how to use the system, explain that you are an AI Document Assistant capable of indexing PDFs (with layout-aware parsing, dense/sparse hybrid search, and cross-encoder reranking), answering complex document questions, comparing specifications, and providing grounded inline page citations.
- Keep conversational answers natural, helpful, and concise.
"""

class LLMClientManager:
    """
    Singleton manager maintaining persistent httpx connection pools with HTTP keep-alive.
    Avoids expensive TCP/TLS handshake latency on every request.
    """
    _groq_client: Optional[AsyncOpenAI] = None
    _shared_http_client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_http_client(cls) -> httpx.AsyncClient:
        if cls._shared_http_client is None or cls._shared_http_client.is_closed:
            cls._shared_http_client = httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=httpx.Timeout(settings.GROQ_TIMEOUT, connect=3.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0)
            )
        return cls._shared_http_client

    @classmethod
    def get_groq_client(cls) -> AsyncOpenAI:
        if cls._groq_client is None:
            http_client = cls.get_http_client()
            cls._groq_client = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url=settings.GROQ_BASE_URL,
                http_client=http_client,
                max_retries=1
            )
        return cls._groq_client


def prepare_context_and_citations(reranked_chunks: List[Dict[str, Any]], max_chunks: int = 6) -> tuple[str, List[CitationBadge]]:
    """
    Context assembler: Expands context window to top 6 chunks for rich synthesis.
    """
    if not reranked_chunks:
        return "NO_RELEVANT_CONTEXT_FOUND", []

    context_pieces = []
    badges = []

    # Selected chunks up to max_chunks
    selected_chunks = reranked_chunks[:max_chunks]

    for idx, chunk_dict in enumerate(selected_chunks):
        citation_id = idx + 1
        chunk = RankedChunk(**chunk_dict)
        doc_id = chunk.document_id
        page = chunk.page_number
        meta = chunk.metadata or {}
        filename = meta.get("filename", "document.pdf")

        block = (
            f"--- Context Block [{citation_id}] ---\n"
            f"Document ID: {doc_id}\n"
            f"Filename: {filename}\n"
            f"Page Number: {page}\n"
            f"Section: {chunk.section_header or 'General'}\n"
            f"Content:\n{chunk.text}\n"
        )
        context_pieces.append(block)

        badges.append(
            CitationBadge(
                citation_id=citation_id,
                chunk_id=chunk.chunk_id,
                document_id=doc_id,
                filename=filename,
                page_number=page,
                section_header=chunk.section_header,
                preview_text=chunk.text[:180] + ("..." if len(chunk.text) > 180 else ""),
                relevance_score=chunk.relevance_score
            )
        )

    return "\n".join(context_pieces), badges


async def generate_response_stream(
    user_query: str,
    route: RouteDecision,
    reranked_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]]
) -> AsyncGenerator[str, None]:
    """
    High-speed asynchronous token streaming generator:
    - Primary: Groq qwen/qwen3.8-27b with persistent connection pool (TTFT < 400ms)
    - Fallback: NVIDIA NIM openai/gpt-oss-20b on timeout/failure
    """
    groq_client = LLMClientManager.get_groq_client()

    try:
        if route == RouteDecision.REJECT:
            yield f"data: {json.dumps({'type': 'token', 'content': 'I am unable to process this request because it violates safety guidelines.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        if route == RouteDecision.CONVERSATIONAL:
            messages = [{"role": "system", "content": CONVERSATIONAL_PROMPT}]
            for turn in chat_history[-4:]:
                if turn.get("role") in ["user", "assistant"]:
                    messages.append({"role": turn["role"], "content": turn.get("content", "")})
            messages.append({"role": "user", "content": user_query})

            stream = await groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                stream=True,
                max_tokens=600,
                temperature=0.3
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # RAG Query route
        context_str, badges = prepare_context_and_citations(reranked_chunks, max_chunks=settings.RERANKER_TOP_K)
        badges_payload = [b.model_dump() for b in badges]
        yield f"data: {json.dumps({'type': 'citations', 'payload': badges_payload})}\n\n"

        if not reranked_chunks or context_str == "NO_RELEVANT_CONTEXT_FOUND":
            fallback_msg = (
                "Based on the provided documents, there is insufficient information to answer this question. "
                "The retrieved passages did not meet the relevance confidence threshold."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': fallback_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        grounding_content = (
            f"### CONTEXT BLOCKS:\n{context_str}\n\n"
            f"### USER QUERY:\n{user_query}\n\n"
            f"Please synthesize a clear, detailed, and complete answer grounded in the context above. "
            f"Cite relevant claims with [Doc: <id>, Page: <page>]."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": grounding_content}
        ]

        # Execute purely on Groq engine
        stream = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            stream=True,
            max_tokens=1500,
            temperature=0.2
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Error during streaming generation: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
