from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    file_size_bytes: int
    total_pages: int
    total_chunks: int

class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    page_number: int
    chunk_index: int
    section_header: Optional[str] = None
    has_table: bool = False
    has_image: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)

class ChunkPayload(BaseModel):
    text: str
    metadata: ChunkMetadata

class IngestResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    file_hash: str
    total_pages: int
    total_chunks: int
    message: str
    is_duplicate: bool = False

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    total_pages: int
    total_chunks: int
    uploaded_at: Optional[str] = None
