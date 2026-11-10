
import json
from pathlib import Path
from statistics import mean

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.ingestion.pipeline import run_ingestion
from src.retrieval.hybrid_search import tokenize, hybrid_search
from src.retrieval.semantic_search import build_search_index
from src.retrieval.reranker import (
    RERANKER_MODEL,
    rerank_documents,
)
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "questions.json"
)


def unique_sources(results):
    """Return unique source filenames in ranking order."""

    sources = []

    for result in results:
        source = result["chunk"].metadata.get("source", "Unknown")
        source_name = Path(str(source)).name

        if source_name not in sources:
            sources.append(source_name)

    return sources


def evaluate_retrieval():
    """Evaluate hybrid retrieval and reranking separately."""

    # 1. Load evaluation questions.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    # 2. Validate the dataset.
    required_fields = {
        "question",
        "relevant_sources",
        "answerable",
    }

    for number, item in enumerate(questions, start=1):
        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Question {number} is missing fields: "
                f"{sorted(missing)}"
            )

        if not isinstance(item["relevant_sources"], list):
            raise ValueError(
                f"Question {number}: relevant_sources "
                "must be a list."
            )

        if not isinstance(item["answerable"], bool):
            raise ValueError(
                f"Question {number}: answerable must be "
                "True or False."
            )

        if item["answerable"] and not item["relevant_sources"]:
            raise ValueError(
                f"Question {number} is marked answerable "
                "but has no relevant sources."
            )

        if not item["answerable"] and item["relevant_sources"]:
            raise ValueError(
                f"Question {number} is marked unanswerable "
                "but has relevant sources."
            )

    answerable_questions = [
        item for item in questions if item["answerable"]
    ]

    unanswerable_questions = [
        item for item in questions if not item["answerable"]
    ]

    if not answerable_questions:
        raise ValueError(
            "At least one answerable question is required."
        )

    # 3. Load and split documents.
    chunks = run_ingestion()

    if not chunks:
        raise ValueError(
            "No document chunks available for evaluation."
        )

    document_names = {
        Path(str(chunk.metadata.get("source", "Unknown"))).name
        for chunk in chunks
    }

    print(f"Documents loaded: {len(document_names)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Answerable questions: {len(answerable_questions)}")
    print(f"Unanswerable questions: {len(unanswerable_questions)}")

    # 4. Build retrieval components once.
    embedding_model, index = build_search_index(chunks)

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    # 5. Load the reranker once.
    reranker = CrossEncoder(RERANKER_MODEL)

    metric_values = {
        "hybrid_recall_1": [],
        "hybrid_recall_2": [],
        "hybrid_mrr": [],
        "hybrid_ndcg_3": [],
        "reranked_recall_1": [],
        "reranked_recall_2": [],
        "reranked_mrr": [],
        "reranked_ndcg_3": [],
    }

    top_source_changes = 0
    top_source_changes_answerable = 0

    unanswerable_diagnostics = []

    # 6. Evaluate every question.
    for item in questions:
        question = item["question"]
        relevant_sources = set(item["relevant_sources"])
        answerable = item["answerable"]

        # Stage A: Hybrid retrieval.
        hybrid_results = hybrid_search(
            question=question,
            chunks=chunks,
            model=embedding_model,
            index=index,
            bm25=bm25,
        )

        # Stage B: CrossEncoder reranking.
        reranked_results = rerank_documents(
            question=question,
            candidates=hybrid_results,
            model=reranker,
        )

        hybrid_sources = unique_sources(hybrid_results)
        reranked_sources = unique_sources(reranked_results)

        # Check whether the top-ranked source changed.
        top_changed = (
            bool(hybrid_sources)
            and bool(reranked_sources)
            and hybrid_sources[0] != reranked_sources[0]
        )

        if top_changed:
            top_source_changes += 1

            if answerable:
                top_source_changes_answerable += 1

        print(f"\nQuestion: {question}")
        print(f"Expected answerable: {answerable}")
        print(f"Hybrid ranking: {hybrid_sources}")
        print(f"Reranked ranking: {reranked_sources}")
        print(f"Top source changed: {top_changed}")

        # Unanswerable questions are diagnosed separately.
        if not answerable:
            unanswerable_diagnostics.append({
                "question": question,
                "hybrid_top": (
                    hybrid_sources[0] if hybrid_sources else "None"
                ),
                "reranked_top": (
                    reranked_sources[0]
                    if reranked_sources
                    else "None"
                ),
            })

            print(
                "Retrieval metrics: skipped "
                "(unanswerable question)"
            )
            continue

        # Stage C: Calculate metrics for answerable questions only.
        relevant = list(relevant_sources)

        current_metrics = {
            "hybrid_recall_1": recall_at_k(
                hybrid_sources, relevant, k=1
            ),
            "hybrid_recall_2": recall_at_k(
                hybrid_sources, relevant, k=2
            ),
            "hybrid_mrr": reciprocal_rank(
                hybrid_sources, relevant
            ),
            "hybrid_ndcg_3": ndcg_at_k(
                hybrid_sources, relevant, k=3
            ),
            "reranked_recall_1": recall_at_k(
                reranked_sources, relevant, k=1
            ),
            "reranked_recall_2": recall_at_k(
                reranked_sources, relevant, k=2
            ),
            "reranked_mrr": reciprocal_rank(
                reranked_sources, relevant
            ),
            "reranked_ndcg_3": ndcg_at_k(
                reranked_sources, relevant, k=3
            ),
        }

        for name, value in current_metrics.items():
            metric_values[name].append(value)

        print(
            f"Hybrid — Recall@1: "
            f"{current_metrics['hybrid_recall_1']:.3f}, "
            f"Recall@2: "
            f"{current_metrics['hybrid_recall_2']:.3f}, "
            f"MRR: {current_metrics['hybrid_mrr']:.3f}, "
            f"nDCG@3: "
            f"{current_metrics['hybrid_ndcg_3']:.3f}"
        )

        print(
            f"Reranked — Recall@1: "
            f"{current_metrics['reranked_recall_1']:.3f}, "
            f"Recall@2: "
            f"{current_metrics['reranked_recall_2']:.3f}, "
            f"MRR: {current_metrics['reranked_mrr']:.3f}, "
            f"nDCG@3: "
            f"{current_metrics['reranked_ndcg_3']:.3f}"
        )

    # 7. Report ranking metrics on answerable questions only.
    metric_labels = {
        "recall_1": "Recall@1",
        "recall_2": "Recall@2",
        "mrr": "MRR",
        "ndcg_3": "nDCG@3",
    }

    print("\n========== ANSWERABLE QUESTION RESULTS ==========")
    print(
        f"Metrics calculated over "
        f"{len(answerable_questions)} answerable questions."
    )

    print("\n--- Hybrid Retrieval ---")

    for metric, label in metric_labels.items():
        values = metric_values[f"hybrid_{metric}"]
        print(f"{label}: {mean(values):.3f}")

    print("\n--- Hybrid Retrieval + Reranking ---")

    for metric, label in metric_labels.items():
        values = metric_values[f"reranked_{metric}"]
        print(f"{label}: {mean(values):.3f}")

    # 8. Report reranking impact.
    print("\n========== RERANKING IMPACT ==========")
    print(
        f"Top source changed across all questions: "
        f"{top_source_changes}/{len(questions)}"
    )
    print(
        f"Top source changed for answerable questions: "
        f"{top_source_changes_answerable}/"
        f"{len(answerable_questions)}"
    )

    # 9. Report unanswerable questions separately.
    print("\n========== UNANSWERABLE QUESTION DIAGNOSTICS ==========")
    print(
        f"Unanswerable questions examined: "
        f"{len(unanswerable_diagnostics)}"
    )

    for result in unanswerable_diagnostics:
        print(f"\nQuestion: {result['question']}")
        print(f"Hybrid top source: {result['hybrid_top']}")
        print(f"Reranked top source: {result['reranked_top']}")
        print(
            "Note: retrieved sources alone do not establish "
            "whether sufficient evidence exists to answer."
        )


if __name__ == "__main__":
    evaluate_retrieval()
