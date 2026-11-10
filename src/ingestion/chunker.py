import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents into smaller overlapping chunks.
    Preserve the original document metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

        identity = (
            chunk.metadata["source"]
            + "\n"
            + chunk.page_content
        )

        chunk.metadata["chunk_uid"] = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:16]

    return chunks


if __name__ == "__main__":
    from .loader import load_documents
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    documents_folder = project_root / "data" / "documents"

    documents = load_documents(documents_folder)
    chunks = split_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print(f"\nChunk ID: {chunk.metadata['chunk_id']}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Text:\n{chunk.page_content}")
