from app.retrieval.hybrid import fuse

def search(bm25_index, dense_index, query: str, k: int = 5, dense_weight: float = 0.5) -> list[dict]:
    bm25_results = bm25_index.search(query, k=k)
    dense_results = dense_index.search(query, k=k)
    ranked = fuse(bm25_results, dense_results, dense_weight=dense_weight)[:k]

    by_id = {doc['id']: doc['text'] for doc in bm25_index.documents}
    return [{'id': doc_id, 'text': by_id[doc_id], 'score': score} for doc_id, score in ranked]