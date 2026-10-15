from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class CitationBadge(BaseModel):
    citation_id: int = Field(description="Citation index e.g., 1, 2, 3 corresponding to [Doc: <id>, Page: <p>]")
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_header: Optional[str] = None
    preview_text: str
    relevance_score: float

class QueryStreamRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    selected_documents: List[str] = Field(default_factory=list)
