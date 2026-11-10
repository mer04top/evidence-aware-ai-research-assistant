VERIFICATION_SYSTEM_PROMPT = """You check one claim against one piece of evidence.

Reply with exactly one word:
SUPPORTED if the evidence states or clearly implies the claim,
CONTRADICTED if the evidence states the opposite,
INSUFFICIENT if the evidence says nothing about the claim.

Rules:
1. Use only the evidence. Do not use outside knowledge.
2. If the claim says that a paper, study or author suggests, reports or claims something, answer SUPPORTED when the evidence says that someone suggests, reports or claims it, even if the evidence does not prove it."""

VERIFICATION_USER_TEMPLATE = """Claim:
{claim}

Evidence:
{evidence}"""