from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from threading import Lock
from fastapi import UploadFile, File
from src.ingestion.upload_pipeline import process_uploaded_document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generation.rag_pipeline import (
    answer_question,
    build_rag_components,
)

uploaded_rag_components = {}
# Store the RAG components so they are loaded only once.
rag_components = None

# Prevent concurrent requests from using the shared models
# at the same time.
rag_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline when the API starts."""

    global rag_components

    print("Starting Advanced RAG API...")

    try:
        rag_components = build_rag_components()
        print("RAG pipeline loaded successfully.")
        yield

    finally:
        rag_components = None
        print("Advanced RAG API stopped.")


app = FastAPI(
    title="Advanced RAG System API",
    description=(
        "Ask questions about your documents using "
        "hybrid retrieval, reranking, and Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    """Request body for the question-answering endpoint."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to ask about the indexed documents.",
    )


class AnswerResponse(BaseModel):
    """Response returned by the question-answering endpoint."""

    question: str
    answer: str


@app.get("/")
def home():
    """Return basic application information."""

    return {
        "message": "Welcome to the Advanced RAG System API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """Check whether the API and RAG pipeline are ready."""

    if rag_components is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    return {
        "status": "healthy",
        "rag_pipeline_loaded": True,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Retrieve evidence and generate an answer."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace.",
        )

    if rag_components is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    (
        chunks,
        embedding_model,
        index,
        bm25,
        reranker,
    ) = rag_components

    try:
        with rag_lock:
            answer = answer_question(
                question=question,
                chunks=chunks,
                embedding_model=embedding_model,
                index=index,
                bm25=bm25,
                reranker=reranker,
            )

        return AnswerResponse(
            question=question,
            answer=answer,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        # Keep internal exception details out of the API response.
        print(f"Question-answering error: {error}")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while answering the question.",
        ) from error

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    try:
        file_bytes = file.file.read()

        components = process_uploaded_document(
            filename=filename,
            file_bytes=file_bytes,
        )

        # Generate an ID for this uploaded document.
        import uuid
        document_id = str(uuid.uuid4())

        uploaded_rag_components[document_id] = components

        chunks = components[0]

        return {
            "message": "Document uploaded and processed successfully.",
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Document upload error: {error}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the uploaded document.",
        ) from error

    finally:
        file.file.close()


from pydantic import BaseModel, Field


class UploadedQuestionRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)


@app.post("/ask-uploaded")
def ask_uploaded_document(request: UploadedQuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace.",
        )

    components = uploaded_rag_components.get(request.document_id)

    if components is None:
        raise HTTPException(
            status_code=404,
            detail="Uploaded document not found. Please upload it again.",
        )

    chunks, embedding_model, index, bm25, reranker = components

    try:
        with rag_lock:
            answer = answer_question(
                question=question,
                chunks=chunks,
                embedding_model=embedding_model,
                index=index,
                bm25=bm25,
                reranker=reranker,
            )

        return {
            "document_id": request.document_id,
            "question": question,
            "answer": answer,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Uploaded-document question error: {error}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while answering the question.",
        ) from error