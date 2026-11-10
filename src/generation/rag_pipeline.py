
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import create_embeddings
from src.retrieval.semantic_search import build_search_index
from src.retrieval.hybrid_search import tokenize, hybrid_search
from src.retrieval.reranker import (
    RERANKER_MODEL,
    rerank_documents,
)
from src.generation.answer_generator import generate_answer


def build_rag_components():
    """Load documents and prepare the retrieval components."""

    print("Loading and processing documents...")
    chunks = run_ingestion()

    if not chunks:
        raise ValueError(
            "No document chunks found. Add documents to data/documents/."
        )

    print(f"Loaded {len(chunks)} document chunks.")

    print("Building semantic search index...")
    embedding_model, index = build_search_index(chunks)

    print("Building keyword search index...")
    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    return chunks, embedding_model, index, bm25, reranker


def answer_question(
    question: str,
    chunks,
    embedding_model,
    index,
    bm25,
    reranker,
) -> str:
    """Retrieve evidence, rerank it, and generate an answer."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve candidate chunks using hybrid search.
    candidates = hybrid_search(
        question=question,
        chunks=chunks,
        model=embedding_model,
        index=index,
        bm25=bm25,
    )

    # Keep the highest-ranked hybrid candidates for reranking.
    candidates = candidates[:5]

    if not candidates:
        return (
            "No relevant evidence was retrieved, "
            "so I cannot provide a reliable answer."
        )

    # Step 2: Rerank candidates using the cross-encoder.
    reranked = rerank_documents(
        question=question,
        candidates=candidates,
        model=reranker,
    )

    # Step 3: Pass the best evidence to Gemini.
    best_documents = [
        item["chunk"]
        for item in reranked[:3]
    ]

    return generate_answer(
        question=question,
        documents=best_documents,
    )


if __name__ == "__main__":
    (
        chunks,
        embedding_model,
        index,
        bm25,
        reranker,
    ) = build_rag_components()

    question = input(
        "\nAsk a question about your documents: "
    )

    answer = answer_question(
        question=question,
        chunks=chunks,
        embedding_model=embedding_model,
        index=index,
        bm25=bm25,
        reranker=reranker,
    )

    print("\n" + "=" * 50)
    print("GENERATED ANSWER")
    print("=" * 50)
    print(answer)
