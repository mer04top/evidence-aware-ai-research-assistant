from app.verification.prompts import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_TEMPLATE

CONFIDENCE_THRESHOLD = 0.75


def parse_llm_verdict(response: str) -> str:
    words = response.strip().upper().split()
    first = words[0].strip('.,:;!') if words else ''
    return {
        'SUPPORTED': 'supported',
        'CONTRADICTED': 'contradicted',
        'INSUFFICIENT': 'insufficient',
    }.get(first, 'insufficient')


def ask_llm(llm, claim: str, evidence_text: str) -> str:
    user_prompt = VERIFICATION_USER_TEMPLATE.format(claim=claim, evidence=evidence_text)
    return parse_llm_verdict(llm.complete(VERIFICATION_SYSTEM_PROMPT, user_prompt))


def verify(nli_model, llm, claim: str, evidence_text: str) -> dict:
    if not evidence_text.strip():
        return {'status': 'insufficient', 'score': 0.0, 'scores': None, 'method': 'none'}

    if nli_model is None:
        status = ask_llm(llm, claim, evidence_text)
        return {'status': status, 'score': None, 'scores': None, 'method': 'llm'}

    scores = nli_model(evidence_text, claim)
    status = max(scores, key=scores.get)
    score = scores[status]

    # NLI may give confident contradictions on unrelated text, so the LLM must confirm them
    if score >= CONFIDENCE_THRESHOLD and status != 'contradicted':
        return {'status': status, 'score': score, 'scores': scores, 'method': 'nli'}

    status = ask_llm(llm, claim, evidence_text)
    return {'status': status, 'score': score, 'scores': scores, 'method': 'llm'}