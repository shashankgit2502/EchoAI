"""
EchoAI Configuration Settings

LLM Provider Configuration:
---------------------------
This module loads LLM provider settings from .env file.
Configure ONE provider at a time by setting the appropriate USE_* flag:
- USE_OLLAMA=true    -> On-Premise Ollama
- USE_OPENROUTER=true -> OpenRouter (current default for development)
- USE_AZURE=true     -> Azure OpenAI (for Azure deployment)
- USE_OPENAI=true    -> Direct OpenAI API

See .env file for detailed configuration options.
"""

import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class LLMSettings(BaseModel):
    """LLM Provider Settings"""

    # Provider flags (only one should be true at a time)
    use_ollama: bool = os.getenv('USE_OLLAMA', 'false').lower() == 'true'
    use_openrouter: bool = os.getenv('USE_OPENROUTER', 'true').lower() == 'true'
    use_azure: bool = os.getenv('USE_AZURE', 'false').lower() == 'true'
    use_openai: bool = os.getenv('USE_OPENAI', 'false').lower() == 'true'

    # Ollama settings
    ollama_base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://10.188.100.131:8004/v1')
    ollama_model: str = os.getenv('OLLAMA_MODEL', 'mistral-nemo:12b-instruct-2407-fp16')

    # OpenRouter settings (current default)
    openrouter_api_key: str = os.getenv(
        'OPENROUTER_API_KEY',
        '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'
    )
    openrouter_base_url: str = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    openrouter_model: str = os.getenv('OPENROUTER_MODEL', 'mistralai/devstral-2512:free')

    # Azure OpenAI settings
    azure_openai_api_key: Optional[str] = os.getenv('AZURE_OPENAI_API_KEY')
    azure_openai_endpoint: Optional[str] = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_deployment: Optional[str] = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    azure_openai_api_version: str = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

    # Direct OpenAI settings
    openai_api_key: Optional[str] = os.getenv('OPENAI_API_KEY')
    openai_model: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    # Default LLM parameters
    default_temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    default_max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '4000'))

    def get_active_provider(self) -> str:
        """Get the currently active LLM provider name."""
        if self.use_azure:
            return 'azure'
        elif self.use_openrouter:
            return 'openrouter'
        elif self.use_ollama:
            return 'ollama'
        elif self.use_openai:
            return 'openai'
        else:
            return 'openrouter'  # Default fallback


class Settings(BaseModel):
    """Application Settings"""

    # App settings
    app_name: str = os.getenv('APP_NAME', 'echo-mermaid-platform')
    service_mode: str = os.getenv('SERVICE_MODE', 'mono')  # mono | micro
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

    # JWT settings
    jwt_secret: str = os.getenv('JWT_SECRET', 'dev-secret-change-me')
    jwt_issuer: str = os.getenv('JWT_ISSUER', 'echo')
    jwt_audience: str = os.getenv('JWT_AUDIENCE', 'echo-clients')

    # LLM settings
    llm: LLMSettings = LLMSettings()


# Global settings instance
settings = Settings()

# Convenience access to LLM settings
llm_settings = settings.llm
