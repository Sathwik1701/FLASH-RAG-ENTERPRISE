from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.layout_parser import ParsedPage
from app.models.document import ChunkMetadata, ChunkPayload

class HierarchicalChunker:
    """
    Optimized structure-aware hierarchical chunking engine.
    Targets 1200 character windows (~250-300 tokens) with 10% overlap to produce
    optimal density and maintain sub-5s vectorization speed across 100-150 pages.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 120
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        pages: List[ParsedPage]
    ) -> List[ChunkPayload]:
        chunks: List[ChunkPayload] = []
        global_chunk_idx = 0

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            current_section_header: Optional[str] = None
            for block in page.blocks:
                if block.get("is_header"):
                    current_section_header = block.get("text")
                    break

            raw_splits = self.text_splitter.split_text(page_text)

            for split_text in raw_splits:
                clean_text = split_text.strip()
                if len(clean_text) < 15:
                    continue

                metadata = ChunkMetadata(
                    document_id=document_id,
                    filename=filename,
                    file_hash=file_hash,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    section_header=current_section_header,
                    has_table=page.has_tables,
                    has_image=page.has_images,
                    extra={"char_count": len(clean_text)}
                )

                chunks.append(
                    ChunkPayload(
                        text=clean_text,
                        metadata=metadata
                    )
                )
                global_chunk_idx += 1

        return chunks
