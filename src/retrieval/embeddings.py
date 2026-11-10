
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document


# A small, widely used model for semantic search
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(
    chunks: list[Document],
) -> tuple[SentenceTransformer, list[list[float]]]:
    """Convert document chunks into numerical vectors."""

    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk.page_content for chunk in chunks]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return model, vectors.tolist()


if __name__ == "__main__":
    from src.ingestion.pipeline import run_ingestion

    chunks = run_ingestion()
    model, vectors = create_embeddings(chunks)

    print(f"\nChunks embedded: {len(vectors)}")
    print(f"Numbers in first embedding: {len(vectors[0])}")
    print(f"First 5 numbers: {vectors[0][:5]}")
