import numpy as np
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, cosine_similarity
from app.retrieval.hybrid import fuse
from tests.fixtures.fake_embedder import FakeEmbedder
from tests.fixtures.retrieval_fixtures import DOCUMENTS


def test_bm25_returns_matching_document():
    results = BM25Index(DOCUMENTS).search('Python', k=2)
    assert results[0][0] in {1, 2}


def test_cosine_similarity():
    scores = cosine_similarity(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_dense_index_uses_embedder():
    embedder = FakeEmbedder({
        'Python is a programming language.': [1, 0],
        'FastAPI is a Python web framework.': [0.9, 0.1],
        'BM25 is a lexical retrieval method.': [0, 1],
        'Python': [1, 0],
    })
    assert DenseIndex(DOCUMENTS, embedder).search('Python', k=1)[0][0] == 1


def test_fusion_uses_bm25_when_weight_is_zero():
    assert fuse([(1, 10.0), (2, 5.0)], [(1, 0.5), (2, 0.9)], dense_weight=0.0)[0][0] == 1


def test_fusion_uses_dense_when_weight_is_one():
    assert fuse([(1, 10.0), (2, 5.0)], [(1, 0.5), (2, 0.9)], dense_weight=1.0)[0][0] == 2

# one test is missing - add later
def test_fusion_rejects_invalid_dense_weight():
    import pytest
    with pytest.raises(ValueError):
        fuse([(1, 10.0)], [(1, 0.5)], dense_weight=5.0)
