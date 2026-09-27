import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

load_dotenv()

LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL",
    "http://127.0.0.1:1234/v1",
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "YOUR_EXACT_LM_STUDIO_MODEL_ID",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

NUM_CTX = int(os.getenv("NUM_CTX", "2048"))

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "solar_support",
)


def build_chat_model(temperature:float=0, response_format=None, need_reasoning=True):
    options = {
        "model": CHAT_MODEL,
        "base_url": LM_STUDIO_BASE_URL,
        "api_key": SecretStr("lm-studio"),
        "temperature": temperature,
        "max_tokens": NUM_CTX,
    }

    if response_format is not None:
        options["response_format"] = response_format

    if not need_reasoning:
        options["reasoning"] = {
            "effort": "none",
            "summary": None,
        }
    else:
        options["reasoning_effort"] = "low"

    return ChatOpenAI(**options)


def build_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LM_STUDIO_BASE_URL,
        api_key=SecretStr("lm-studio"),
        check_embedding_ctx_length=False,
    )


def build_chroma_client():
    import chromadb

    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )