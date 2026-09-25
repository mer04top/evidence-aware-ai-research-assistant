import os

from dotenv import load_dotenv
from openai import OpenAI

from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, SentenceTransformerEmbedder
from app.retrieval.hybrid import fuse
from tests.fixtures.retrieval_fixtures import DOCUMENTS

load_dotenv()

question = 'What is the capital of Pakistan?'

bm25_results = BM25Index(DOCUMENTS).search(question, k=3)
dense_results = DenseIndex(DOCUMENTS, SentenceTransformerEmbedder()).search(question, k=3)
ranked = fuse(bm25_results, dense_results, dense_weight=0.5)

by_id = {doc['id']: doc['text'] for doc in DOCUMENTS}
passages = [by_id[doc_id] for doc_id, _ in ranked]

context = '\n'.join(passages)

prompt = f"""Answer the question using only the context below.

Context:
{context}

Question:
{question}"""

client = OpenAI(
    api_key=os.getenv('LLM_API_KEY'),
    base_url=os.getenv('LLM_BASE_URL'),
)

response = client.chat.completions.create(
    model=os.getenv('LLM_MODEL_NAME'),
    temperature=0,
    messages=[
        {'role': 'system', 'content': 'Answer only using the supplied context.'},
        {'role': 'user', 'content': prompt},
    ],
)

print(response.choices[0].message.content)

# answered a question docs shouldn't be able to ("what's the capital of Pakistan")
# should say it didn't know
# need to fix that in the actual prompt once this moves into app/