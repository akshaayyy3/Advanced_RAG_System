
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.generation.rag_pipeline import answer_question


def make_document():
    """Create a sample document for testing."""

    return Document(
        page_content="A server may overheat when cooling fans fail.",
        metadata={
            "source": "sample.txt",
            "chunk_id": 0,
        },
    )


@patch("src.generation.rag_pipeline.generate_answer")
@patch("src.generation.rag_pipeline.rerank_documents")
@patch("src.generation.rag_pipeline.hybrid_search")
def test_rag_pipeline_connects_all_stages(
    mock_hybrid_search,
    mock_rerank_documents,
    mock_generate_answer,
):
    document = make_document()

    mock_hybrid_search.return_value = [
        {"chunk": document, "score": 0.03}
    ]

    mock_rerank_documents.return_value = [
        {"chunk": document, "rerank_score": 0.95}
    ]

    mock_generate_answer.return_value = (
        "A server may overheat when cooling fans fail "
        "[sample.txt, Chunk ID: 0]."
    )

    answer = answer_question(
        question="Why is my server overheating?",
        chunks=[document],
        embedding_model=MagicMock(),
        index=MagicMock(),
        bm25=MagicMock(),
        reranker=MagicMock(),
    )

    assert "cooling fans fail" in answer
    mock_hybrid_search.assert_called_once()
    mock_rerank_documents.assert_called_once()
    mock_generate_answer.assert_called_once()


def test_rag_pipeline_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(
            question="   ",
            chunks=[],
            embedding_model=None,
            index=None,
            bm25=None,
            reranker=None,
        )


@patch("src.generation.rag_pipeline.hybrid_search")
def test_rag_pipeline_handles_no_retrieved_candidates(
    mock_hybrid_search,
):
    mock_hybrid_search.return_value = []

    answer = answer_question(
        question="Who manufactured this server?",
        chunks=[],
        embedding_model=MagicMock(),
        index=MagicMock(),
        bm25=MagicMock(),
        reranker=MagicMock(),
    )

    assert "no relevant evidence was retrieved" in answer.lower()


@patch("src.generation.rag_pipeline.generate_answer")
@patch("src.generation.rag_pipeline.rerank_documents")
@patch("src.generation.rag_pipeline.hybrid_search")
def test_rag_pipeline_returns_abstention_when_evidence_is_insufficient(
    mock_hybrid_search,
    mock_rerank_documents,
    mock_generate_answer,
):
    """Check that an abstention from the generator reaches the user."""

    document = make_document()

    mock_hybrid_search.return_value = [
        {"chunk": document, "score": 0.03}
    ]

    mock_rerank_documents.return_value = [
        {"chunk": document, "rerank_score": 0.95}
    ]

    mock_generate_answer.return_value = (
        "The available documents do not specify the server's warranty period."
    )

    answer = answer_question(
        question="What is the warranty period of the server?",
        chunks=[document],
        embedding_model=MagicMock(),
        index=MagicMock(),
        bm25=MagicMock(),
        reranker=MagicMock(),
    )

    assert "do not specify" in answer.lower()
    mock_generate_answer.assert_called_once()


@patch("src.generation.rag_pipeline.generate_answer")
@patch("src.generation.rag_pipeline.rerank_documents")
@patch("src.generation.rag_pipeline.hybrid_search")
def test_rag_pipeline_handles_empty_reranker_results(
    mock_hybrid_search,
    mock_rerank_documents,
    mock_generate_answer,
):
    """Check that empty reranker output cannot cause an index error."""

    document = make_document()

    mock_hybrid_search.return_value = [
        {"chunk": document, "score": 0.03}
    ]

    mock_rerank_documents.return_value = []
    mock_generate_answer.return_value = (
        "I could not find sufficient evidence to answer reliably."
    )

    answer = answer_question(
        question="What is the server's warranty period?",
        chunks=[document],
        embedding_model=MagicMock(),
        index=MagicMock(),
        bm25=MagicMock(),
        reranker=MagicMock(),
    )

    # The pipeline currently passes an empty document list to the generator.
    assert "could not find sufficient evidence" in answer.lower()
    mock_generate_answer.assert_called_once_with(
        question="What is the server's warranty period?",
        documents=[],
    )
