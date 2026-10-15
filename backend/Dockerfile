# ==============================================================================
# Production Dockerfile for Multimodal Hybrid RAG API
# Optimized for Render Web Services & Google Cloud Run
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies required for layout analysis & native build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache local ONNX embedding weights and FlashRank models at image build time
# (Eliminates runtime cold-start download delay on Render)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/tmp/flashrank')"

# Copy application code
COPY . .

# Expose port (Render automatically maps $PORT)
EXPOSE 8000

# Start Uvicorn server with production keep-alive settings for SSE streaming
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --timeout-keep-alive 65"]
