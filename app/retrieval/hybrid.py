def min_max_normalize(results: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not results:
        return []
    scores = [score for _, score in results]
    low, high = min(scores), max(scores)
    if high == low:
        return [(doc_id, 1.0) for doc_id, _ in results]
    return [(doc_id, (score - low) / (high - low)) for doc_id, score in results]


def fuse(bm25_results, dense_results, dense_weight=0.5):
    if not 0 <= dense_weight <= 1:
        raise ValueError('dense_weight must be between 0 and 1')
    bm25 = dict(min_max_normalize(bm25_results))
    dense = dict(min_max_normalize(dense_results))
    ids = set(bm25) | set(dense)
    results = []
    for doc_id in ids:
        score = (1 - dense_weight) * bm25.get(doc_id, 0.0) + dense_weight * dense.get(doc_id, 0.0)
        results.append((doc_id, score))
    return sorted(results, key=lambda item: item[1], reverse=True)
