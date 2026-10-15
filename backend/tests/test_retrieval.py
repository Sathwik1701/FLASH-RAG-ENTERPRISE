import pytest
from app.models.intent import RetrievalFilters, ContentTypeFilter
from app.rag.retrieval_node import build_qdrant_filter
from qdrant_client import models

def test_build_qdrant_filter_with_doc_ids():
    filters = RetrievalFilters(
        document_ids=["doc_1", "doc_2"],
        content_type=ContentTypeFilter.TABLE
    )
    q_filter = build_qdrant_filter(filters)
    assert q_filter is not None
    assert len(q_filter.must) == 2
    assert isinstance(q_filter.must[0], models.FieldCondition)
    assert q_filter.must[0].key == "metadata.document_id"
    assert q_filter.must[1].key == "metadata.has_table"

def test_build_qdrant_filter_empty():
    filters = RetrievalFilters(
        document_ids=[],
        content_type=ContentTypeFilter.ANY
    )
    q_filter = build_qdrant_filter(filters)
    assert q_filter is None
