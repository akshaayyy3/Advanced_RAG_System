# Advanced RAG System

An advanced Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from uploaded PDF and TXT documents.

The system combines semantic search, keyword search, hybrid retrieval, reranking, and Gemini-based answer generation to provide relevant, evidence-based responses.

## Features

- **Document Ingestion:** Load PDF and TXT documents.
- **Text Chunking:** Split documents into smaller sections for effective retrieval.
- **Semantic Search:** Use Sentence Transformers embeddings and FAISS.
- **Keyword Search:** Use BM25 to find relevant terms.
- **Hybrid Retrieval:** Combine semantic and keyword search using Reciprocal Rank Fusion (RRF).
- **Reranking:** Improve result ordering using a CrossEncoder model.
- **Answer Generation:** Generate answers using Google Gemini.
- **Evidence-Based Responses:** Use retrieved document content to support answers.
- **Source Citations:** Display references to retrieved documents.
- **Abstention:** Avoid generating a definitive answer when sufficient evidence is unavailable.
- **Document Upload:** Support user-uploaded PDF and TXT files.
- **API:** Provide application functionality through FastAPI.
- **Evaluation:** Measure retrieval performance and test system behaviour.

## System Architecture

1. Document loading
2. Text chunking
3. Embedding generation
4. FAISS semantic retrieval
5. BM25 keyword retrieval
6. Hybrid search using RRF
7. CrossEncoder reranking
8. Gemini answer generation
9. Source-based response

## Technologies Used

- Python
- LangChain
- Sentence Transformers
- FAISS
- BM25
- Google Gemini
- FastAPI
- HTML
- CSS
- JavaScript
- Pytest

## Evaluation

The retrieval pipeline was evaluated using 15 questions, including answerable and unanswerable questions.

| Metric | Result |
|---|---:|
| Recall@1 | 0.944 |
| Recall@2 | 1.000 |
| MRR | 1.000 |
| nDCG@3 | 1.000 |

These results are from a small evaluation dataset and should not be interpreted as a guarantee of performance on larger or unseen datasets.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Advanced_RAG_System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root and configure your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Never upload your `.env` file or expose your API key publicly.

## Running the Application

Start the FastAPI application using the entry point configured in your project.

For example, if your application entry point is `src.api.main`:

```bash
uvicorn src.api.main:app --reload
```

Open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Testing

Run the test suite:

```bash
pytest
```

## Limitations

- Retrieval evaluation uses a relatively small dataset.
- Generation evaluation uses heuristic checks and does not guarantee factual correctness.
- Results depend on the quality and relevance of uploaded documents.
- The system may abstain when the retrieved evidence is insufficient.

## Future Improvements

- Expand the evaluation dataset.
- Improve generation evaluation.
- Test performance on larger document collections.
- Deploy the application for online access.

## Author
AKSHAY AGARWAL
MSc Artificial Intelligence Project
