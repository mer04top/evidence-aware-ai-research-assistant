from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.service import search
from tests.fixtures.fake_embedder import FakeEmbedder
from tests.fixtures.retrieval_fixtures import DOCUMENTS

EMBEDDER = FakeEmbedder({
    'Python is a programming language.': [1, 0],
    'FastAPI is a Python web framework.': [0.9, 0.1],
    'BM25 is a lexical retrieval method.': [0, 1],
    'Python': [1, 0],
    'FastAPI is built on Python.': [0.9, 0.1],
})


def build_indexes():
    return BM25Index(DOCUMENTS), DenseIndex(DOCUMENTS, EMBEDDER)


def test_search_returns_full_passages_not_just_ids():
    bm25_index, dense_index = build_indexes()

    results = search(bm25_index, dense_index, 'Python', k=2)

    assert results[0]['id'] in {1, 2}
    assert 'text' in results[0]
    assert 'score' in results[0]


def test_search_respects_k():
    bm25_index, dense_index = build_indexes()

    results = search(bm25_index, dense_index, 'Python', k=1)

    assert len(results) == 1


def test_search_works_the_same_for_a_claim_as_a_question():
    # search() does not distinguish a question from a claim
    bm25_index, dense_index = build_indexes()

    results = search(bm25_index, dense_index, 'FastAPI is built on Python.', k=2)

    ids = [r['id'] for r in results]
    assert 2 in ids  # the FastAPI document should show up