import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL_NAME")

        if not api_key:
            raise ValueError("LLM_API_KEY is not set")
        if not self.model:
            raise ValueError("LLM_MODEL_NAME is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL"),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content
        return answer or ""
