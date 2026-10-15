import pytest
from app.models.retrieval import RetrievedChunk
from app.models.rerank import RankedChunk
from app.services.reranker import sigmoid, RerankerService
from app.rag.reranker_node import CrossEncoderReranker

def test_sigmoid_math():
    assert round(sigmoid(0.0), 2) == 0.5
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01

@pytest.mark.asyncio
async def test_empty_candidates():
    result = await CrossEncoderReranker.rerank(
        query="What are the terms?",
        candidate_chunks=[],
        top_n=5,
        score_threshold=0.35
    )
    assert result == []

@pytest.mark.asyncio
async def test_reranker_threshold_filtering(monkeypatch):
    # Mock reranker service to return controlled scores
    monkeypatch.setattr(
        "app.services.reranker.reranker_service.score_pairs",
        lambda query, passages: [0.85, 0.20, 0.92]
    )

    candidates = [
        RetrievedChunk(chunk_id="1", document_id="doc1", text="Relevant chunk 1"),
        RetrievedChunk(chunk_id="2", document_id="doc1", text="Low score chunk 2"),
        RetrievedChunk(chunk_id="3", document_id="doc1", text="Highly relevant chunk 3")
    ]

    result = await CrossEncoderReranker.rerank(
        query="Test query",
        candidate_chunks=candidates,
        top_n=5,
        score_threshold=0.35
    )

    # Chunk 2 should be filtered out because 0.20 < 0.35
    assert len(result) == 2
    assert result[0].chunk_id == "3"  # 0.92
    assert result[1].chunk_id == "1"  # 0.85
    assert result[0].relevance_score == 0.92
