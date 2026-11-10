
import faiss
import numpy as np

from src.ingestion.pipeline import run_ingestion
from src.retrieval.embeddings import create_embeddings


def build_search_index(chunks):
    """Store chunk embeddings in a searchable FAISS index."""

    model, vectors = create_embeddings(chunks)

    embeddings = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def search_documents(
    question: str,
    chunks,
    model,
    index,
    top_k: int = 2,
):
    """Find the most relevant chunks for a question."""

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    )

    query_vector = np.asarray(query_vector, dtype="float32")

    top_k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        results.append({
            "score": float(score),
            "chunk": chunks[int(chunk_index)],
        })

    return results


if __name__ == "__main__":
    chunks = run_ingestion()
    model, index = build_search_index(chunks)

    question = "python -m pip install rank-bm25"

    results = search_documents(
        question=question,
        chunks=chunks,
        model=model,
        index=index,
    )

    print(f"\nQuestion: {question}")

    for result in results:
        chunk = result["chunk"]

        print(f"\nSimilarity score: {result['score']:.4f}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Text:\n{chunk.page_content}")
