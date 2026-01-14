"""
Application Configuration
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    APP_NAME: str = "Workflow Orchestrator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API Keys (loaded from environment)
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] ="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"


    # LLM Configuration
    DEFAULT_LLM_MODEL: str = "mistralai/devstral-2512:free"
    DEFAULT_LLM_TEMPERATURE: float = 0.7
    DEFAULT_LLM_MAX_TOKENS: int = 4000

    # Storage
    STORAGE_BASE_PATH: Path = Path(__file__).parent.parent.parent / "app" / "storage"
    WORKFLOWS_PATH: Path = STORAGE_BASE_PATH / "workflows"
    AGENTS_PATH: Path = STORAGE_BASE_PATH / "agents"

    # Workflow Lifecycle
    ENABLE_AUTO_VERSIONING: bool = True
    MAX_WORKFLOW_VERSIONS: int = 50

    # Execution
    DEFAULT_EXECUTION_TIMEOUT: int = 300  # seconds
    ENABLE_CHECKPOINTING: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Using lru_cache ensures settings are loaded once and reused
    """
    return Settings()
