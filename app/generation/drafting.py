from app.generation.prompts import (
    DRAFTING_SYSTEM_PROMPT,
    DRAFTING_USER_TEMPLATE,
    INSUFFICIENT_CONTEXT,
)


def draft_answer(llm, question: str, passages: list[dict], max_passages: int = 10) -> dict:
    passages = passages[:max_passages]

    if not passages:
        return {'question': question, 'answer': '', 'passages': [], 'sufficient': False}

    passage_text = '\n'.join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    user_prompt = DRAFTING_USER_TEMPLATE.format(question=question, passages=passage_text)

    answer = llm.complete(DRAFTING_SYSTEM_PROMPT, user_prompt).strip()
    sufficient = INSUFFICIENT_CONTEXT not in answer

    return {
        'question': question,
        'answer': answer if sufficient else '',
        'passages': passages,
        'sufficient': sufficient,
    }