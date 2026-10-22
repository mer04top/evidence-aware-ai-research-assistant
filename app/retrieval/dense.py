from typing import Protocol
import numpy as np

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings)

def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    vectors = np.asarray(vectors, dtype=float)

    if vectors.size == 0:
        return np.array([], dtype=float)

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)

    if query_norm == 0:
        return np.zeros(len(vectors))

    safe_norms = np.where(vector_norms == 0, 1.0, vector_norms)
    return (vectors @ query) / (safe_norms * query_norm)

class DenseIndex:
    def __init__(self, documents: list[dict], embedder: Embedder):
        self.documents = documents
        self.embedder = embedder

        if documents:
            texts = [doc["text"] for doc in documents]
            self.vectors = np.asarray(embedder.encode(texts), dtype=float)
        else:
            self.vectors = np.empty((0, 0), dtype=float)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if k < 0:
            raise ValueError("k must not be negative")
        if not self.documents or not query.strip() or k == 0:
            return []

        query_vector = self.embedder.encode([query])[0]
        scores = cosine_similarity(query_vector, self.vectors)
        order = np.argsort(scores)[::-1][:k]

        return [
            (self.documents[i]["id"], float(scores[i]))
            for i in order
        ]
