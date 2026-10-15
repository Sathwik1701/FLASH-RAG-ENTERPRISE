from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class RankedChunk(BaseModel):
    chunk_id: str = Field(description="Unique Qdrant point identifier.")
    document_id: str = Field(description="Originating document identifier.")
    text: str = Field(description="Clean raw text content of the chunk.")
    page_number: int = Field(default=1, description="Page number of the chunk.")
    chunk_index: int = Field(default=0, description="Ordinal index of the chunk in the document.")
    section_header: Optional[str] = Field(default=None, description="Upstream section/clause heading.")
    has_table: bool = Field(default=False, description="Flag indicating if chunk contains tabular data.")
    has_image: bool = Field(default=False, description="Flag indicating if chunk is associated with an image.")
    relevance_score: float = Field(description="Sigmoid-normalized cross-encoder relevance score in [0.0, 1.0].")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata payload.")
