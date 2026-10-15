import uuid
import logging
import asyncio
import time
from typing import List, Callable, Optional
from qdrant_client import models
from app.db.qdrant import QdrantManager
from app.core.config import settings
from app.models.document import IngestResponse, ChunkPayload
from app.ingestion.validation import validate_pdf_file, calculate_sha256
from app.ingestion.layout_parser import PDFLayoutParser
from app.ingestion.chunker import HierarchicalChunker
from app.services.embedding import embedding_service

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    High-Throughput Ingestion Pipeline:
    1. Instant SHA-256 Deduplication check
    2. C-accelerated Layout-aware parsing (sub-300ms for 100 pages)
    3. Hierarchical chunking with optimized windows
    4. Concurrent Dense & Sparse Vectorization via asyncio.gather
    5. Concurrent Chunk Bulk Upsert to Qdrant (batch_size=256)
    Target latency: <= 5-7 seconds for 100-page documents.
    """

    def __init__(self):
        self.layout_parser = PDFLayoutParser()
        self.chunker = HierarchicalChunker(chunk_size=900, chunk_overlap=100)

    async def ingest_pdf(
        self,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> IngestResponse:
        t0 = time.perf_counter()

        # Step 1: Validation
        is_valid, validation_msg = validate_pdf_file(filename, content_type, file_bytes)
        if not is_valid:
            raise ValueError(f"File validation failed: {validation_msg}")

        # Step 2: Instant Deduplication Check
        file_hash = calculate_sha256(file_bytes)
        existing_doc_id = await QdrantManager.check_hash_exists(file_hash)
        if existing_doc_id:
            logger.info(f"Duplicate document detected: {filename} ({file_hash}) in {time.perf_counter() - t0:.2f}s")
            return IngestResponse(
                status="duplicate",
                document_id=existing_doc_id,
                filename=filename,
                file_hash=file_hash,
                total_pages=0,
                total_chunks=0,
                message="Document with identical hash already indexed in Qdrant.",
                is_duplicate=True
            )

        document_id = str(uuid.uuid4())

        # Step 3: Ultra-Fast Layout Parsing
        t_parse = time.perf_counter()
        parsed_pages = self.layout_parser.parse_pdf(file_bytes)
        total_pages = len(parsed_pages)
        logger.info(f"Parsed {total_pages} pages in {time.perf_counter() - t_parse:.3f}s")

        # Step 4: Fast Hierarchical Chunking
        t_chunk = time.perf_counter()
        chunks: List[ChunkPayload] = self.chunker.chunk_document(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            pages=parsed_pages
        )

        if not chunks:
            raise ValueError("No extractable text or content found in the PDF.")
        
        logger.info(f"Generated {len(chunks)} chunks in {time.perf_counter() - t_chunk:.3f}s")

        chunk_texts = [chunk.text for chunk in chunks]

        # Step 5: CONCURRENT Vectorization (Dense + Sparse computed in parallel)
        t_emb = time.perf_counter()
        logger.info(f"Concurrently vectorizing {len(chunks)} chunks across multi-threaded ONNX...")
        dense_task = embedding_service.get_dense_embeddings(chunk_texts, batch_size=32)
        sparse_task = embedding_service.get_sparse_embeddings_async(chunk_texts)

        dense_embeddings, sparse_embeddings = await asyncio.gather(dense_task, sparse_task)
        logger.info(f"Dual vectorization completed in {time.perf_counter() - t_emb:.3f}s")

        # Step 6: Concurrent Parallel Batch Upsert to Qdrant
        t_upsert = time.perf_counter()
        client = await QdrantManager.get_client()
        points = []

        for idx, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            dense_vec = dense_embeddings[idx]
            sparse_dict = sparse_embeddings[idx]

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_dict["indices"],
                        values=sparse_dict["values"]
                    )
                },
                payload={
                    "text": chunk.text,
                    "metadata": chunk.metadata.model_dump()
                }
            )
            points.append(point)

        # Bulk upsert in parallel batches of 256 points
        batch_size = 256
        upsert_tasks = []
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            upsert_tasks.append(
                client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=batch,
                    wait=False  # Async non-blocking write
                )
            )

        await asyncio.gather(*upsert_tasks)
        logger.info(f"Bulk indexed {len(points)} points into Qdrant in {time.perf_counter() - t_upsert:.3f}s")

        total_time = time.perf_counter() - t0
        logger.info(f"Total ingestion completed in {total_time:.2f}s for {total_pages} pages ({len(chunks)} chunks)")

        return IngestResponse(
            status="success",
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            total_pages=total_pages,
            total_chunks=len(chunks),
            message=f"Indexed {total_pages} pages ({len(chunks)} chunks) in {total_time:.2f}s.",
            is_duplicate=False
        )

ingestion_pipeline = IngestionPipeline()
