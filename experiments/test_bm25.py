from app.retrieval.bm25 import BM25Index

documents = [
    {'id': 1, 'text': 'PostgreSQL stores relational data.'},
    {'id': 2, 'text': 'BM25 ranks documents using term statistics.'},
]
print(BM25Index(documents).search('BM25 term statistics'))
