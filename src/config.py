import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(...)
    openai_model: str = Field(default="gpt-4-turbo-preview")
    embedding_model: str = Field(default="text-embedding-3-small")

    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    database_url: str = Field(default="sqlite:///./ecommerce_chatbot.db")
    database_echo: bool = Field(default=False)

    vector_db_path: str = Field(default="./vector_db")
    vector_collection_name: str = Field(default="products")
    vector_search_k: int = Field(default=5)

    max_chat_history: int = Field(default=50)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    max_function_calls: int = Field(default=10)
    function_timeout: int = Field(default=30)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def validate_environment() -> None:
    required_vars: list[str] = ["OPENAI_API_KEY"]
    missing_vars: list[str] = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")


def get_openai_client():
    try:
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key)
    except ImportError as e:
        raise ImportError(
            "OpenAI package not installed. Please install with: pip install openai"
        ) from e


def get_langfuse_client():
    if not settings.langfuse_configured:
        return None

    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError as e:
        raise ImportError(
            "Langfuse package not installed. Please install with: pip install langfuse"
        ) from e
