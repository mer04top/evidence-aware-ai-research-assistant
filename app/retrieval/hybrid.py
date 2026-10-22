def min_max_normalize(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not results:
        return []

    scores = [score for _, score in results]
    low = min(scores)
    high = max(scores)

    if high == low:
        return [(doc_id, 1.0) for doc_id, _ in results]

    normalized = []
    for doc_id, score in results:
        normalized_score = (score - low) / (high - low)
        normalized.append((doc_id, normalized_score))
    return normalized

def fuse(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    dense_weight: float = 0.5
) -> list[tuple[str, float]]:
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between 0 and 1")

    bm25_scores = dict(min_max_normalize(bm25_results))
    dense_scores = dict(min_max_normalize(dense_results))

    all_ids = set(bm25_scores) | set(dense_scores)
    results = []

    for doc_id in all_ids:
        bm25_score = bm25_scores.get(doc_id, 0.0)
        dense_score = dense_scores.get(doc_id, 0.0)
        score = ((1 - dense_weight) * bm25_score + dense_weight * dense_score)
        results.append((doc_id, score))

    return sorted(
        results,
        key=lambda item: (-item[1], item[0])
    )
