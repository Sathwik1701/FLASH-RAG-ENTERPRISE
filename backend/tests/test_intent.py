import pytest
import asyncio
from app.models.intent import RouteDecision, ContentTypeFilter, IntentAnalysisOutput
from app.models.state import RAGWorkflowState
from app.rag.intent_node import analyze_intent_node

@pytest.mark.asyncio
async def test_fallback_when_api_key_empty():
    state = RAGWorkflowState(
        user_query="Can you summarize the safety regulations?",
        chat_history=[],
        selected_documents=["doc_123"]
    )
    # Without OPENAI_API_KEY set, it should gracefully fall back to default RAG_QUERY
    result = await analyze_intent_node(state)
    assert "intent" in result
    intent: IntentAnalysisOutput = result["intent"]
    assert intent.route in [RouteDecision.RAG_QUERY, RouteDecision.CONVERSATIONAL]
    assert "doc_123" in intent.retrieval_filters.document_ids
