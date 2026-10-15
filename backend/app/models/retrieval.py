from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    chunk_id: str = Field(description="Unique Qdrant point identifier.")
    document_id: str = Field(description="Document ID originating the chunk.")
    text: str = Field(description="Raw text content of the retrieved passage.")
    page_number: int = Field(default=1, description="Page number where the chunk is located.")
    chunk_index: int = Field(default=0, description="Ordinal index of the chunk in the document.")
    section_header: Optional[str] = Field(default=None, description="Upstream section/clause heading.")
    has_table: bool = Field(default=False, description="Flag indicating if chunk contains tabular data.")
    has_image: bool = Field(default=False, description="Flag indicating if chunk is associated with an image.")
    score: float = Field(default=0.0, description="RRF or hybrid similarity score.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete raw metadata payload.")
