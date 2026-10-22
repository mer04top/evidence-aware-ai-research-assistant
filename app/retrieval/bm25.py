from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return text.lower().split()

class BM25Index:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.tokens = [tokenize(doc["text"]) for doc in documents]
        self.index = BM25Okapi(self.tokens) if self.tokens else None

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if k < 0:
            raise ValueError("k must not be negative")
        if not self.documents or not query.strip() or k == 0:
            return []

        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

        return [
            (self.documents[i]["id"], float(score))
            for i, score in ranked[:k]
        ]
