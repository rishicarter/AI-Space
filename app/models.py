from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
Your name is FastAPI bot and you are a helpful chatbot responsible for
teaching FastAPI to your users. Always respond in simple text and be concise
"""


def load_text_model():
    openai_client = OpenAI(
        base_url=os.getenv("OPENROUTER_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    return openai_client


def generate_text(client: OpenAI, prompt: str, temperature: float = 0.7):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = client.responses.create(
        model=os.getenv("MODEL"),
        temperature=temperature,
        top_p=0.95,
        max_output_tokens=256,
        input=messages,
        reasoning={"effort": "none"}
    )

    return response.output_text
