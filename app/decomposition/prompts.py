DECOMPOSITION_SYSTEM_PROMPT = """Split the answer into atomic claims, each one a
single independently checkable fact. Do not add information that isn't in the
answer.

Write each claim as plain text that makes sense on its own:
- Remove citation markers such as [1] or "Passage [3]".
- Remove formatting such as bullets, bold text and LaTeX.
- Replace words like "this", "it" and "both" with what they refer to.

Return one claim per line and nothing else."""

DECOMPOSITION_USER_TEMPLATE = """Answer:
{answer}"""