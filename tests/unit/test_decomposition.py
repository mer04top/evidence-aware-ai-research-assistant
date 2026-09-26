from app.decomposition.service import decompose
from tests.fixtures.fake_llm import FakeLLM


def test_decompose_splits_into_claims():
    llm = FakeLLM(['Python is a programming language.\nPython was created in 1991.'])

    claims = decompose(llm, 'Python is a programming language, created in 1991.')

    assert claims == [
        'Python is a programming language.',
        'Python was created in 1991.',
    ]


def test_decompose_strips_numbering():
    llm = FakeLLM(['1. Python is a language.\n2. Python was created in 1991.'])

    claims = decompose(llm, 'some answer')

    assert claims == ['Python is a language.', 'Python was created in 1991.']


def test_decompose_strips_bullets():
    llm = FakeLLM(['- Python is a language.\n* Python was created in 1991.'])

    claims = decompose(llm, 'some answer')

    assert claims == ['Python is a language.', 'Python was created in 1991.']


def test_decompose_skips_blank_lines():
    llm = FakeLLM(['Python is a language.\n\n\nPython was created in 1991.'])

    claims = decompose(llm, 'some answer')

    assert len(claims) == 2


def test_empty_answer_skips_llm_call():
    llm = FakeLLM([])

    claims = decompose(llm, '')

    assert claims == []
    assert llm.calls == []