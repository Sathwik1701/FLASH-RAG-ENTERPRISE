from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.models.intent import IntentAnalysisOutput

class MessageItem(BaseModel):
    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message body")

class RAGWorkflowState(BaseModel):
    # Inputs
    user_query: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    selected_documents: List[str] = Field(default_factory=list)

    # Intent Analysis Layer Outputs
    intent: Optional[IntentAnalysisOutput] = None

    # Retrieval & Reranker Outputs
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: List[Dict[str, Any]] = Field(default_factory=list)

    # Generation Outputs
    generated_response: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
