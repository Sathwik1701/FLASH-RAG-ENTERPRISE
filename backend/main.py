import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.qdrant import QdrantManager
from app.api.documents import router as documents_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize Qdrant hybrid collection
    logger.info("Initializing vector database collection...")
    try:
        await QdrantManager.init_collection()
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant at {settings.QDRANT_URL} during startup: {e}")
        logger.warning("Ensure Qdrant is running via Docker ('docker compose up -d') or cloud endpoint.")
    yield
    # Shutdown
    await QdrantManager.close()
    logger.info("Closed Qdrant connections.")

app = FastAPI(
    title="Multimodal Hybrid RAG API",
    version="1.0.0",
    description="Enterprise-grade RAG with Layout-aware Parsing, Dense & Sparse Retrieval, Cross-Encoder Reranking, and SSE Streaming",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(documents_router)
from app.api.query import router as query_router
app.include_router(query_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Multimodal RAG Orchestrator"}
