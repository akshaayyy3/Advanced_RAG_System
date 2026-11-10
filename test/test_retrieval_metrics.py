
import math

from src.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


def test_recall_at_k():
    relevant_ids = [2, 3]
    retrieved_ids = [1, 2, 3]

    result = recall_at_k(retrieved_ids, relevant_ids, k=2)

    assert result == 0.5


def test_reciprocal_rank():
    relevant_ids = [3]
    retrieved_ids = [1, 2, 3, 4]

    result = reciprocal_rank(retrieved_ids, relevant_ids)

    assert abs(result - 1 / 3) < 1e-9


def test_ndcg_at_k():
    relevant_ids = [2]
    retrieved_ids = [1, 2, 3]

    result = ndcg_at_k(retrieved_ids, relevant_ids, k=3)

    expected = 1 / math.log2(3)

    assert abs(result - expected) < 1e-9


def test_recall_when_no_relevant_documents():
    result = recall_at_k([1, 2], [], k=2)

    assert result == 0.0


def test_reciprocal_rank_when_no_match():
    result = reciprocal_rank([1, 2, 3], [4])

    assert result == 0.0


def test_ndcg_when_no_relevant_documents():
    result = ndcg_at_k([1, 2, 3], [], k=3)

    assert result == 0.0
