import json

from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, SentenceTransformerEmbedder
from app.retrieval.service import search

with open('data/corpus.json') as f:
    documents = json.load(f)

print(f'loaded {len(documents)} chunks from data/corpus.json')

bm25_index = BM25Index(documents)
dense_index = DenseIndex(documents, SentenceTransformerEmbedder())

question = input('question: ')

# for r in search(bm25_index, dense_index, question, k=3):
#     print(round(r['score'], 3), r['id'], '-', r['text'][:150])
#     # print(r)

results = search(bm25_index, dense_index, question, k=3)

for r in results:
    print()
    print('score:', round(r['score'], 3))
    print('id:', r['id'])
    print('text:', r['text'][:300])

# nothing in BM25Index, DenseIndex, fuse, or search() changed
# source wrong???? | fixed
