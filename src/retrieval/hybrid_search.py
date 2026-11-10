
import re

from rank_bm25 import BM25Okapi

from src.ingestion.pipeline import run_ingestion
from src.retrieval.semantic_search import (
    build_search_index,
    search_documents,
)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase words."""
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    rank_constant: int = 60,
) -> list[dict]:
    """Combine rankings without comparing incompatible raw scores."""

    fused = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]
            chunk_id = chunk.metadata["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk": chunk,
                    "score": 0.0,
                }

            fused[chunk_id]["score"] += (
                1 / (rank_constant + rank)
            )

    return sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )


def hybrid_search(question: str, chunks, model, index, bm25):
    """Combine BM25 keyword search and semantic search."""

    # 1. Search using keywords
    tokenized_question = tokenize(question)
    keyword_scores = bm25.get_scores(tokenized_question)

    keyword_order = sorted(
        range(len(chunks)),
        key=lambda i: keyword_scores[i],
        reverse=True,
    )

    keyword_results = [
        {"chunk": chunks[i]}
        for i in keyword_order[:len(chunks)]
    ]

    # 2. Search using semantic similarity
    semantic_results = search_documents(
        question=question,
        chunks=chunks,
        model=model,
        index=index,
        top_k=len(chunks),
    )

    # 3. Combine both rankings using RRF
    return reciprocal_rank_fusion(
        [keyword_results, semantic_results]
    )


if __name__ == "__main__":
    chunks = run_ingestion()

    model, index = build_search_index(chunks)

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    question = "Why is my machine getting too hot?"

    results = hybrid_search(
        question,
        chunks,
        model,
        index,
        bm25,
    )

    print(f"\nQuestion: {question}")

    for rank, result in enumerate(results[:2], start=1):
        chunk = result["chunk"]

        print(f"\nRank: {rank}")
        print(f"RRF score: {result['score']:.5f}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Text:\n{chunk.page_content}")
