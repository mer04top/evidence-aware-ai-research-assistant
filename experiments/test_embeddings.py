from app.retrieval.dense import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder()
print(embedder.encode(['Python is a programming language.', 'Python is used for software development.']).shape)
