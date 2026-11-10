
from pathlib import Path

from langchain_core.documents import Document

from .loader import load_documents
from .chunker import split_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_FOLDER = PROJECT_ROOT / "data" / "documents"


def run_ingestion() -> list[Document]:
    """Load documents and split them into chunks."""

    documents = load_documents(DOCUMENTS_FOLDER)

    if not documents:
        print("No documents found.")
        return []

    chunks = split_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    chunks = run_ingestion()

    for chunk in chunks:
        print(f"\nChunk ID: {chunk.metadata['chunk_id']}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Text:\n{chunk.page_content}")
