
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from src.ingestion.pipeline import run_ingestion
from src.retrieval.semantic_search import build_search_index
from src.retrieval.hybrid_search import tokenize, hybrid_search


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_documents(
    question: str,
    candidates: list[dict],
    model: CrossEncoder,
    min_score: float | None = None,
) -> list[dict]:
    """Rerank chunks and optionally filter weak relevance scores."""

    if not candidates:
        return []

    pairs = [
        (question, item["chunk"].page_content)
        for item in candidates
    ]

    scores = model.predict(pairs)

    reranked = []

    for item, score in zip(candidates, scores):
        reranked.append({
            "chunk": item["chunk"],
            "rerank_score": float(score),
        })

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    # Apply a threshold only when one is explicitly provided.
    if min_score is not None:
        reranked = [
            item
            for item in reranked
            if item["rerank_score"] >= min_score
        ]

    return reranked


if __name__ == "__main__":
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

    questions = [
        "Why might a database connection fail?",
        "How can I protect a server against unauthorised access?",
        "What is the warranty period of the server?",
    ]

    for question in questions:
        candidates = hybrid_search(
            question,
            chunks,
            embedding_model,
            index,
            bm25,
        )

        results = rerank_documents(
            question=question,
            candidates=candidates,
            model=reranker,
        )

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Candidates reranked: {len(results)}")

        if not results:
            print("No relevant candidates were retrieved.")
            continue

        for rank, result in enumerate(results[:3], start=1):
            chunk = result["chunk"]

            print(f"\nRank: {rank}")
            print(
                f"Reranker score: "
                f"{result['rerank_score']:.4f}"
            )
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Chunk ID: {chunk.metadata.get('chunk_id', 'Unknown')}")
            print(f"Text: {chunk.page_content[:200]}")
