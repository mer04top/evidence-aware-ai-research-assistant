from app.generation.drafting import draft_answer
from tests.fixtures.fake_llm import FakeLLM
from tests.fixtures.retrieval_fixtures import DOCUMENTS


def test_draft_answer_uses_llm_response():
    llm = FakeLLM(['Python is a programming language.'])

    result = draft_answer(llm, 'What is Python?', DOCUMENTS[:1])

    assert result['sufficient'] is True
    assert result['answer'] == 'Python is a programming language.'


def test_passages_reach_the_prompt():
    llm = FakeLLM(['some answer'])

    draft_answer(llm, 'What is BM25?', DOCUMENTS)

    _system, user_prompt = llm.calls[0]
    for doc in DOCUMENTS:
        assert doc['text'] in user_prompt


def test_no_passages_skips_the_llm_call():
    llm = FakeLLM([])

    result = draft_answer(llm, 'anything', [])

    assert result['sufficient'] is False
    assert result['answer'] == ''
    assert llm.calls == []


def test_insufficient_context_is_detected():
    llm = FakeLLM(['INSUFFICIENT_CONTEXT'])

    result = draft_answer(llm, 'What is the capital of France?', DOCUMENTS[:1])

    assert result['sufficient'] is False
    assert result['answer'] == ''


def test_max_passages_is_respected():
    llm = FakeLLM(['answer'])

    result = draft_answer(llm, 'q', DOCUMENTS, max_passages=1)

    assert len(result['passages']) == 1