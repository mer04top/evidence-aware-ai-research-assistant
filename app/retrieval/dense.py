from typing import Protocol
import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    qnorm = np.linalg.norm(query)
    vnorms = np.linalg.norm(vectors, axis=1)
    if qnorm == 0:
        return np.zeros(len(vectors))
    safe = np.where(vnorms == 0, 1.0, vnorms)
    return (vectors @ query) / (safe * qnorm)


class DenseIndex:
    def __init__(self, documents: list[dict], embedder: Embedder):
        self.documents = documents
        self.embedder = embedder
        self.vectors = np.asarray(embedder.encode([d['text'] for d in documents]), dtype=float)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = cosine_similarity(self.embedder.encode([query])[0], self.vectors)
        order = np.argsort(scores)[::-1][:k]
        return [(self.documents[i]['id'], float(scores[i])) for i in order]
