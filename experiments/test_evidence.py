from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, SentenceTransformerEmbedder
from app.retrieval.hybrid import fuse
from tests.fixtures.retrieval_fixtures import DOCUMENTS

bm25_index = BM25Index(DOCUMENTS)
dense_index = DenseIndex(DOCUMENTS, SentenceTransformerEmbedder())

by_id = {doc['id']: doc['text'] for doc in DOCUMENTS}

claims = [
    'Python is a programming language.',
    # 'FastAPI is built on Python.',
    'Python was created in 1991.',
    'Midhat is a student of SZABIST.',
    'Midhat is in her final year.',
    'Midhat codes in Python.',
]

for claim in claims:
    bm25_results = bm25_index.search(claim, k=2)
    dense_results = dense_index.search(claim, k=2)
    ranked = fuse(bm25_results, dense_results, dense_weight=0.5)

    print(claim)
    for doc_id, score in ranked:
        print(f'  {round(score, 3)}  {by_id[doc_id]}')