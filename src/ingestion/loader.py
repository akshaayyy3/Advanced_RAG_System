
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_FOLDER = PROJECT_ROOT / "data" / "documents"


def load_documents(folder_path: Path) -> list[Document]:
    """Load supported text and PDF documents."""

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Document folder not found: {folder_path}"
        )

    documents = []

    for file_path in sorted(folder_path.iterdir()):

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8-sig")

            if not text.strip():
                print(f"Skipping empty file: {file_path.name}")
                continue

            loaded_docs = [
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_type": "txt",
                    },
                )
            ]

        elif suffix == ".pdf":
            loaded_docs = PyPDFLoader(str(file_path)).load()

            for doc in loaded_docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["file_type"] = "pdf"

        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        documents.extend(loaded_docs)

    return documents


if __name__ == "__main__":
    docs = load_documents(DOCUMENTS_FOLDER)

    print(f"\nDocuments loaded: {len(docs)}")

    for index, doc in enumerate(docs, start=1):
        print(f"\n--- Document/Page {index} ---")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Preview:\n{doc.page_content[:250]}")
