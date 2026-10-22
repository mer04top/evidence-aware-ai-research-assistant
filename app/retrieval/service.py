from app.retrieval.hybrid import fuse

def search(bm25_index, dense_index, query: str, k: int = 5, dense_weight: float = 0.5) -> list[dict]:
    if k < 0:
        raise ValueError("k must not be negative")
    if k == 0 or not query.strip():
        return []

    bm25_results = bm25_index.search(query, k=k)
    dense_results = dense_index.search(query, k=k)

    ranked = fuse(bm25_results, dense_results, dense_weight=dense_weight)[:k]
    documents_by_id = {}

    for index in (bm25_index, dense_index):
        for doc in index.documents:
            documents_by_id[doc["id"]] = doc
    results = []
    for doc_id, score in ranked:
        doc = documents_by_id.get(doc_id)
        if doc is None:
            continue
        results.append({
            "id": doc_id,
            "text": doc["text"],
            "score": score
        })
    return results
