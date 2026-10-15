import re
import logging
from typing import Dict, Any, List, Optional
from app.models.intent import IntentAnalysisOutput, RouteDecision, RetrievalFilters, ContentTypeFilter
from app.models.state import RAGWorkflowState

logger = logging.getLogger(__name__)

# Compile high-performance regex patterns for zero-latency deterministic classification
GREETINGS_PATTERN = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|howdy|sup|hola)\b[\s!\.\?]*$",
    re.IGNORECASE
)
COURTESY_PATTERN = re.compile(
    r"^(?:thanks|thank\s+you|thx|cool|got\s+it|ok|okay|bye|goodbye|see\s+ya)\b[\s!\.\?]*$",
    re.IGNORECASE
)
CAPABILITY_PATTERN = re.compile(
    r"^(?:what\s+can\s+you\s+do|who\s+are\s+you|what\s+are\s+you|how\s+(?:can\s+you|to)\s+help(?:\s+me)?|what\s+is\s+your\s+purpose|help|help\s+me|tell\s+me\s+about\s+yourself)\b[\s!\.\?]*$",
    re.IGNORECASE
)
REJECT_PATTERN = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous\s+instructions|system\s+prompt|jailbreak|bypass\s+filters|dan\s+mode)",
    re.IGNORECASE
)

# Coreference indicators that suggest a follow-up query
COREFERENCE_PATTERN = re.compile(
    r"\b(?:it|they|them|its|their|this|that|these|those|previous|above|mentioned)\b",
    re.IGNORECASE
)

def evaluate_fast_intent(
    query: str,
    chat_history: List[Dict[str, str]],
    selected_docs: List[str]
) -> IntentAnalysisOutput:
    """
    Sub-millisecond dual-track deterministic intent classifier.
    Eliminates the serial 16-second cloud LLM call from the critical path entirely.
    """
    cleaned_query = query.strip()

    # 1. Security / Malicious Guardrail Check (< 0.1ms)
    if REJECT_PATTERN.search(cleaned_query):
        return IntentAnalysisOutput(
            route=RouteDecision.REJECT,
            standalone_query=cleaned_query,
            retrieval_filters=RetrievalFilters(document_ids=selected_docs),
            reasoning="Deterministic guardrail flag: prompt injection or forbidden instruction detected."
        )

    # 2. Conversational Chit-Chat & Capabilities Check (< 0.1ms)
    if GREETINGS_PATTERN.match(cleaned_query) or COURTESY_PATTERN.match(cleaned_query) or CAPABILITY_PATTERN.match(cleaned_query):
        return IntentAnalysisOutput(
            route=RouteDecision.CONVERSATIONAL,
            standalone_query=cleaned_query,
            retrieval_filters=RetrievalFilters(document_ids=selected_docs),
            reasoning="Deterministic rule: conversational greeting, courtesy, or capability inquiry."
        )

    # 3. Contextual Coreference Expansion via Chat History
    standalone_query = cleaned_query
    if chat_history and COREFERENCE_PATTERN.search(cleaned_query):
        # Extract keywords from the last user question for context grounding
        last_turns = [t.get("content", "") for t in chat_history if t.get("role") == "user"]
        if last_turns:
            prior_topic = last_turns[-1].strip()
            # If current query is very short (e.g. "why?", "explain more", "what about its safety?"), enrich it
            if len(cleaned_query.split()) <= 6:
                standalone_query = f"{prior_topic} - {cleaned_query}"

    return IntentAnalysisOutput(
        route=RouteDecision.RAG_QUERY,
        standalone_query=standalone_query,
        retrieval_filters=RetrievalFilters(document_ids=selected_docs),
        reasoning="Dual-track optimistic router: routed directly to parallel hybrid retrieval."
    )

async def analyze_intent_node(state: RAGWorkflowState) -> Dict[str, Any]:
    """
    Non-blocking, sub-millisecond Intent Analysis Node.
    """
    intent = evaluate_fast_intent(
        query=state.user_query,
        chat_history=state.chat_history or [],
        selected_docs=state.selected_documents or []
    )
    return {"intent": intent}
