"""
Application configuration loaded from environment variables.
Copy .env.example to .env and fill in your keys before running.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- LLM provider ----
    LLM_PROVIDER: str = "openrouter" #anthropic

    OPENROUTER_API_KEY: str = ""
    # Free vision-capable model good for OCR tasks
    OPENROUTER_MODEL: str = "google/gemma-4-31b-it:free" #qwen/qwen2.5-vl-72b-instruct:free
    # Fallback text-only free model (no image support needed for PDF text)
    OPENROUTER_TEXT_MODEL: str = "google/gemma-4-31b-it:free" #"meta-llama/llama-3.3-70b-instruct:free"

    # Anthropic (prod)
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "anthropic/claude-3-haiku"

    # ---- App ----
    APP_TITLE: str = "OCR Pipeline API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
