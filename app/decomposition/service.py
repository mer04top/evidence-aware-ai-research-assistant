import re

from app.decomposition.prompts import DECOMPOSITION_SYSTEM_PROMPT, DECOMPOSITION_USER_TEMPLATE


def clean_claim_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r'^\d+[\.\)]\s*', '', line)  # "1. " / "1) "
    line = re.sub(r'^[-*]\s*', '', line)  # "- " / "* "
    return line.strip()


def decompose(llm, answer_text: str) -> list[str]:
    if not answer_text:
        return []

    user_prompt = DECOMPOSITION_USER_TEMPLATE.format(answer=answer_text)
    response = llm.complete(DECOMPOSITION_SYSTEM_PROMPT, user_prompt)

    claims = [clean_claim_line(line) for line in response.splitlines()]
    return [c for c in claims if c]