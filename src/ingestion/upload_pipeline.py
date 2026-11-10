
from pathlib import Path
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retrieval.embeddings import create_embeddings
from src.retrieval.semantic_search import build_search_index
from src.retrieval.hybrid_search import tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.retrieval.reranker import RERANKER_MODEL


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def process_uploaded_document(filename: str, file_bytes: bytes):
    """Process one uploaded PDF or TXT and build its retrieval components."""

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name)

        if extension == ".pdf":
            loader = PyPDFLoader(str(temp_path))
        else:
            loader = TextLoader(
                str(temp_path),
                encoding="utf-8",
            )

        documents = loader.load()

        if not any(
            document.page_content.strip()
            for document in documents
        ):
            raise ValueError(
                "No readable text was found in the uploaded document."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "The uploaded document could not be split into chunks."
            )

        # Attach the original filename to every chunk for citations.
        for chunk in chunks:
            chunk.metadata["source"] = Path(filename).name

        embedding_model, index = build_search_index(chunks)

        tokenized_chunks = [
            tokenize(chunk.page_content)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_chunks)

        reranker = CrossEncoder(RERANKER_MODEL)

        return chunks, embedding_model, index, bm25, reranker

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
