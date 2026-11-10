
import json
from pathlib import Path

from src.generation.rag_pipeline import (
    answer_question,
    build_rag_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "questions.json"
)

RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "generation_results.json"
)


def load_json(path: Path):
    """Load JSON data from a file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_results(results: dict):
    """Save results immediately to prevent losing completed answers."""

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            list(results.values()),
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_existing_results() -> dict:
    """Load old or new evaluation results into a dictionary."""

    if not RESULTS_PATH.exists():
        return {}

    saved_results = load_json(RESULTS_PATH)

    if isinstance(saved_results, list):
        return {
            item["question"]: item
            for item in saved_results
            if isinstance(item, dict) and "question" in item
        }

    if isinstance(saved_results, dict):
        return saved_results

    raise ValueError(
        "Unexpected format in generation_results.json."
    )


def detect_abstention(answer: str) -> bool:
    """
    Detect common phrases indicating insufficient evidence.

    This is a basic phrase-based heuristic, not a guarantee
    that an answer is factually grounded.
    """

    phrases = [
        "cannot provide a reliable answer",
        "could not find relevant evidence",
        "not enough information",
        "not provided in the documents",
        "not provided in the evidence",
        "does not contain",
        "cannot determine",
        "unable to determine",
        "there is no information",
        "there is no mention",
        "not mentioned",
        "not specified",
        "not stated",
        "no information about",
        "information is not available",
        "not available in the provided evidence",
        "not available in the documents",
        "cannot be determined from the provided evidence",
        "not included in the documents",
        "not included in the provided evidence",
        "the documents do not mention",
        "the evidence does not mention",
        "the provided evidence does not state",
        "the provided documents do not state",
        "not found in the documents",
        "not found in the provided evidence",
        "the evidence does not provide",
        "the documents do not provide",
        "the provided evidence does not contain",
    ]

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in phrases
    )


def check_answer_behavior(answer: str, answerable: bool) -> tuple:
    """Return abstention detection and basic behavior check."""

    abstained = detect_abstention(answer)

    if answerable:
        check_passed = not abstained
    else:
        check_passed = abstained

    return abstained, check_passed


def refresh_saved_checks(questions: list, results: dict):
    """
    Recalculate behavior checks for saved answers.

    This function does not call Gemini or rebuild the RAG pipeline.
    """

    updated = 0
    passed = 0
    reviewed = 0

    for item in questions:
        question = item["question"]
        result = results.get(question)

        # Skip missing answers and saved errors.
        if not isinstance(result, dict) or "answer" not in result:
            continue

        answer = result["answer"]
        answerable = item["answerable"]

        abstained, check_passed = check_answer_behavior(
            answer,
            answerable,
        )

        result["expected_answerable"] = answerable
        result["relevant_sources"] = item["relevant_sources"]
        result["abstention_detected"] = abstained
        result["basic_behavior_check"] = check_passed

        updated += 1

        if check_passed:
            passed += 1
        else:
            reviewed += 1

        print(f"\nRechecking: {question}")
        print(f"Abstention detected: {abstained}")
        print(
            "Basic behavior check: "
            + ("PASS" if check_passed else "REVIEW")
        )

    if updated:
        save_results(results)

    print("\n" + "-" * 65)
    print("SAVED ANSWER RECHECK")
    print("-" * 65)
    print(f"Saved answers rechecked: {updated}")
    print(f"Behavior checks passed: {passed}")
    print(f"Questions requiring review: {reviewed}")
    print("No Gemini calls were needed for this recheck.")


def is_quota_error(error: Exception) -> bool:
    """Identify API quota and rate-limit errors."""

    error_text = (
        f"{type(error).__name__}: {error}"
    ).lower()

    quota_indicators = [
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "429",
    ]

    return any(
        indicator in error_text
        for indicator in quota_indicators
    )


def evaluate_generation():
    """Evaluate generation while preserving completed results."""

    questions = load_json(DATASET_PATH)

    if not questions:
        raise ValueError(
            "The evaluation dataset is empty."
        )

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

        if not isinstance(item["answerable"], bool):
            raise ValueError(
                f"Question {number}: 'answerable' must be "
                "a boolean."
            )

        if not isinstance(item["relevant_sources"], list):
            raise ValueError(
                f"Question {number}: 'relevant_sources' "
                "must be a list."
            )

    results = load_existing_results()

    # First, recheck all previously saved answers without an API call.
    refresh_saved_checks(questions, results)

    # Evaluate only questions with missing answers or saved errors.
    pending_questions = [
        item
        for item in questions
        if (
            item["question"] not in results
            or "error" in results[item["question"]]
            or "answer" not in results[item["question"]]
        )
    ]

    completed_count = sum(
        "answer" in results.get(item["question"], {})
        for item in questions
    )

    print(f"\nTotal questions: {len(questions)}")
    print(f"Previously completed: {completed_count}")
    print(f"Questions to evaluate: {len(pending_questions)}")

    if not pending_questions:
        print("\nAll questions already have saved answers.")
        print(f"Results file: {RESULTS_PATH}")
        print("No generation API calls were required.")
        return

    # Build RAG components only if new answers are needed.
    (
        chunks,
        embedding_model,
        index,
        bm25,
        reranker,
    ) = build_rag_components()

    for number, item in enumerate(pending_questions, start=1):
        question = item["question"]
        answerable = item["answerable"]

        print("\n" + "=" * 65)
        print(f"QUESTION {number}/{len(pending_questions)}")
        print(question)
        print(f"Expected answerable: {answerable}")

        try:
            answer = answer_question(
                question=question,
                chunks=chunks,
                embedding_model=embedding_model,
                index=index,
                bm25=bm25,
                reranker=reranker,
            )

            abstained, check_passed = check_answer_behavior(
                answer,
                answerable,
            )

            results[question] = {
                "question": question,
                "expected_answerable": answerable,
                "relevant_sources": item["relevant_sources"],
                "answer": answer,
                "abstention_detected": abstained,
                "basic_behavior_check": check_passed,
            }

            print("\nGenerated answer:")
            print(answer)
            print(f"\nAbstention detected: {abstained}")
            print(
                "Basic behavior check: "
                + ("PASS" if check_passed else "REVIEW")
            )

            # Save every successful answer immediately.
            save_results(results)

        except Exception as error:
            error_message = (
                f"{type(error).__name__}: {error}"
            )

            print(f"\nEvaluation error: {error_message}")

            results[question] = {
                "question": question,
                "expected_answerable": answerable,
                "relevant_sources": item["relevant_sources"],
                "error": error_message,
            }

            save_results(results)

            if is_quota_error(error):
                print(
                    "\nGemini API quota or rate limit reached."
                )
                print(
                    "Stopping to avoid repeated failed API requests."
                )
                print(
                    "Retry after checking the quota reset time."
                )
                break

    # Calculate final summary using answers saved for this dataset.
    evaluated_results = [
        results[item["question"]]
        for item in questions
        if (
            item["question"] in results
            and "answer" in results[item["question"]]
        )
    ]

    passed = sum(
        result.get("basic_behavior_check") is True
        for result in evaluated_results
    )

    reviewed = sum(
        result.get("basic_behavior_check") is False
        for result in evaluated_results
    )

    errors = sum(
        "error" in results.get(item["question"], {})
        for item in questions
    )

    print("\n" + "=" * 65)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Questions in dataset: {len(questions)}")
    print(f"Answers generated and saved: {len(evaluated_results)}")
    print(f"Basic behavior checks passed: {passed}")
    print(f"Questions requiring review: {reviewed}")
    print(f"Questions with errors: {errors}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate_generation()
