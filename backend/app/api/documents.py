from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.models.document import IngestResponse, DocumentInfo
from app.ingestion.pipeline import ingestion_pipeline
from app.db.qdrant import QdrantManager
from app.core.config import settings

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", response_model=IngestResponse)
@router.post("/upload/", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a PDF document:
    Validates file, checks SHA-256 hash, extracts layout, chunks, embeds (Dense+Sparse), and indexes to Qdrant.
    """
    try:
        file_bytes = await file.read()
        response = await ingestion_pipeline.ingest_pdf(
            filename=file.filename or "unknown.pdf",
            content_type=file.content_type or "application/pdf",
            file_bytes=file_bytes
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    """
    List all indexed documents in Qdrant with chunk counts and page statistics.
    """
    try:
        client = await QdrantManager.get_client()
        # Scroll to aggregate document metadata
        records, _ = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        docs_map = {}
        for record in records:
            meta = record.payload.get("metadata", {})
            doc_id = meta.get("document_id")
            if not doc_id:
                continue

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "file_hash": meta.get("file_hash", ""),
                    "total_pages": meta.get("page_number", 1),
                    "total_chunks": 1
                }
            else:
                docs_map[doc_id]["total_chunks"] += 1
                docs_map[doc_id]["total_pages"] = max(docs_map[doc_id]["total_pages"], meta.get("page_number", 1))

        return [DocumentInfo(**data) for data in docs_map.values()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
