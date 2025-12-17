"""Configuration settings for the AI chatbot"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model"
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./ecommerce_chatbot.db", description="Database URL"
    )
    database_echo: bool = Field(default=False, description="SQLAlchemy echo mode")

    # Vector Database Configuration
    vector_db_path: str = Field(default="./vector_db", description="Vector DB path")
    vector_collection_name: str = Field(
        default="products", description="Vector collection name"
    )
    vector_search_k: int = Field(default=5, description="Number of search results")

    # Application Configuration
    max_chat_history: int = Field(default=50, description="Maximum chat history")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")

    # Function Calling Configuration
    max_function_calls: int = Field(
        default=10, description="Maximum function calls per conversation"
    )
    function_timeout: int = Field(default=30, description="Function timeout in seconds")

    class Config:
        """Pydantic settings configuration"""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


# Global settings instance
settings = get_settings()


def validate_environment() -> None:
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")


def get_openai_client():
    """Get OpenAI client instance"""
    try:
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key)
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Please install with: pip install openai"
        )