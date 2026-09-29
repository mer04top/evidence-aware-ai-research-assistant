import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv('LLM_API_KEY'),
    base_url=os.getenv('LLM_BASE_URL'),
)

answer = 'Python is a programming language, created in 1991. Midhat is student of SZABIST, in her final year, who codes in Python'

prompt = f"""Split the answer below into atomic claims, one per line.

Answer:
{answer}"""

response = client.chat.completions.create(
    model=os.getenv('LLM_MODEL_NAME'),
    temperature=0,
    messages=[{'role': 'user', 'content': prompt}],
)

print(response.choices[0].message.content)