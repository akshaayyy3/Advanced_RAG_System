
def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int],
    k: int = 2,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: list[int],
) -> float:
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int],
    k: int = 3,
) -> float:
    relevant = set(relevant_ids)

    dcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank, chunk_id in enumerate(
            retrieved_ids[:k], start=1
        )
        if chunk_id in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0
