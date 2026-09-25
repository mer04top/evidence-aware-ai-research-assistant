import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv('LLM_API_KEY'),
    base_url=os.getenv('LLM_BASE_URL'),
)

response = client.chat.completions.create(
    model=os.getenv('LLM_MODEL_NAME'),
    temperature=0,
    messages=[
        {'role': 'user', 'content': 'Say OK and nothing else.'},
    ],
)

print(response.choices[0].message.content)