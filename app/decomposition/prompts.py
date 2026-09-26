DECOMPOSITION_SYSTEM_PROMPT = """Split the answer into atomic claims - each one a single
independently checkable fact. Do not add information that isn't in the answer.
One claim per line, nothing else."""

DECOMPOSITION_USER_TEMPLATE = """Answer:
{answer}"""