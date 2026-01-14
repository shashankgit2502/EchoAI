"""
LLM Provider Abstraction
Multi-provider LLM support with unified interface, cost tracking, and availability checks
"""
import json
import os  # For Azure deployment environment variables
from typing import Optional, Dict, Any, List, AsyncIterator
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# For Azure deployment - uncomment the line below
# from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.constants import LLMModel

logger = get_logger(__name__)


# ============================================================================
# MODEL METADATA
# ============================================================================

@dataclass
class ModelMetadata:
    """
    LLM model metadata from catalog
    """
    id: str
    name: str
    provider: str
    family: str
    tier: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_vision: bool
    supports_streaming: bool
    cost_per_million_input_tokens: float
    cost_per_million_output_tokens: float
    recommended_for: List[str]
    description: str

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for given token counts

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.cost_per_million_input_tokens
        output_cost = (output_tokens / 1_000_000) * self.cost_per_million_output_tokens
        return input_cost + output_cost


@dataclass
class ProviderMetadata:
    """
    LLM provider metadata
    """
    id: str
    name: str
    api_base_url: str
    requires_api_key: bool
    api_key_env_var: str
    supported_features: List[str]
    rate_limits: Dict[str, int]


# ============================================================================
# MODEL CATALOG
# ============================================================================

class ModelCatalog:
    """
    Registry of available LLM models

    Loads from llm_models.json and provides lookup methods
    """

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize model catalog

        Args:
            catalog_path: Path to llm_models.json (defaults to schemas/llm_models.json)
        """
        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent / "schemas" / "llm_models.json"

        self.catalog_path = catalog_path
        self._models: Dict[str, ModelMetadata] = {}
        self._providers: Dict[str, ProviderMetadata] = {}
        self._load_catalog()

    def _load_catalog(self):
        """Load models and providers from catalog"""
        try:
            with open(self.catalog_path, 'r') as f:
                catalog = json.load(f)

            # Load models
            for model_data in catalog.get("models", []):
                metadata = ModelMetadata(
                    id=model_data["id"],
                    name=model_data["name"],
                    provider=model_data["provider"],
                    family=model_data["family"],
                    tier=model_data["tier"],
                    context_window=model_data["context_window"],
                    max_output_tokens=model_data["max_output_tokens"],
                    supports_tools=model_data["supports_tools"],
                    supports_vision=model_data["supports_vision"],
                    supports_streaming=model_data["supports_streaming"],
                    cost_per_million_input_tokens=model_data["cost_per_million_input_tokens"],
                    cost_per_million_output_tokens=model_data["cost_per_million_output_tokens"],
                    recommended_for=model_data["recommended_for"],
                    description=model_data["description"]
                )
                self._models[metadata.id] = metadata

            # Load providers
            for provider_data in catalog.get("providers", []):
                metadata = ProviderMetadata(
                    id=provider_data["id"],
                    name=provider_data["name"],
                    api_base_url=provider_data["api_base_url"],
                    requires_api_key=provider_data["requires_api_key"],
                    api_key_env_var=provider_data["api_key_env_var"],
                    supported_features=provider_data["supported_features"],
                    rate_limits=provider_data["rate_limits"]
                )
                self._providers[metadata.id] = metadata

            logger.info(
                f"Loaded model catalog: {len(self._models)} models, "
                f"{len(self._providers)} providers"
            )

        except Exception as e:
            logger.error(f"Failed to load model catalog: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load LLM model catalog: {e}")

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        """Normalize model IDs for lookup (e.g., strip OpenRouter prefix)."""
        if model_id.startswith("openrouter/"):
            return model_id[len("openrouter/"):]
        return model_id

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        normalized_id = self._normalize_model_id(model_id)
        return self._models.get(normalized_id)

    def get_provider(self, provider_id: str) -> Optional[ProviderMetadata]:
        """Get provider metadata by ID"""
        return self._providers.get(provider_id)

    def get_models_by_provider(self, provider: str) -> List[ModelMetadata]:
        """Get all models for a provider"""
        return [m for m in self._models.values() if m.provider == provider]

    def get_models_by_tier(self, tier: str) -> List[ModelMetadata]:
        """Get all models in a tier (premium, standard, fast)"""
        return [m for m in self._models.values() if m.tier == tier]

    def list_all_models(self) -> List[ModelMetadata]:
        """Get all available models"""
        return list(self._models.values())

    def list_all_providers(self) -> List[ProviderMetadata]:
        """Get all available providers"""
        return list(self._providers.values())


# ============================================================================
# LLM PROVIDER
# ============================================================================

class LLMProvider:
    """
    Unified LLM provider abstraction

    Supports multiple providers (Anthropic, OpenAI) with:
    - Unified invoke interface
    - Model availability checking
    - Cost tracking
    - Token counting
    - Streaming support
    - Error handling

    Usage:
        provider = LLMProvider()

        # Check model availability
        is_available = await provider.check_availability("claude-sonnet-4-5-20250929")

        # Invoke model
        response = await provider.ainvoke(
            model="claude-sonnet-4-5-20250929",
            messages=[HumanMessage(content="Hello")],
            temperature=0.7,
            max_tokens=1000
        )

        # Get cost estimate
        cost = provider.estimate_cost(
            model="claude-sonnet-4-5-20250929",
            input_tokens=1000,
            output_tokens=500
        )
    """

    def __init__(self, catalog: Optional[ModelCatalog] = None):
        """
        Initialize LLM provider

        Args:
            catalog: Model catalog (defaults to global catalog)
        """
        self.settings = get_settings()
        self.catalog = catalog or ModelCatalog()
        self._model_cache: Dict[str, BaseChatModel] = {}

    def _get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for provider

        Args:
            provider: Provider ID (anthropic, openai, openrouter, onprem)

        Returns:
            API key or None if not configured
        """
        provider_meta = self.catalog.get_provider(provider)
        if not provider_meta:
            return None

        # On-prem doesn't require API key
        if provider == "onprem":
            return "ollama"  # Default API key for Ollama

        if provider == "anthropic":
            return self.settings.ANTHROPIC_API_KEY
        elif provider == "openai":
            return self.settings.OPENAI_API_KEY
        elif provider == "openrouter":
            return getattr(self.settings, 'OPENROUTER_API_KEY', None) or "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"
        else:
            return None

    def _create_model(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> BaseChatModel:
        """
        Create LangChain model instance

        Args:
            model_id: Model ID
            temperature: Temperature (0.0-2.0)
            max_tokens: Max output tokens
            **kwargs: Additional model parameters

        Returns:
            LangChain chat model

        Raises:
            ValueError: If model not found or API key missing
        """
        # Get model metadata
        model_meta = self.catalog.get_model(model_id)
        if not model_meta:
            raise ValueError(f"Unknown model: {model_id}")

        normalized_model_id = self.catalog._normalize_model_id(model_id)

        # Get API key
        api_key = self._get_api_key(model_meta.provider)
        if not api_key:
            raise ValueError(
                f"Missing API key for {model_meta.provider}. "
                f"Set {self.catalog.get_provider(model_meta.provider).api_key_env_var} "
                "environment variable."
            )

        # Create provider-specific model
        if model_meta.provider == "anthropic":
            return ChatAnthropic(
                model=normalized_model_id,
                anthropic_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        elif model_meta.provider == "openai":
            # For Azure deployment - uncomment this block
            # return AzureChatOpenAI(
            #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     **kwargs
            # )

            # For local/standard OpenAI - comment this when deploying to Azure
            return ChatOpenAI(
                model=normalized_model_id,
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        elif model_meta.provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            provider_meta = self.catalog.get_provider("openrouter")

            # For Azure deployment - uncomment this block
            # return AzureChatOpenAI(
            #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     **kwargs
            # )

            # For local/OpenRouter - comment this when deploying to Azure
            return ChatOpenAI(
                base_url=provider_meta.api_base_url,
                openai_api_key=api_key,
                model=normalized_model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        elif model_meta.provider == "onprem":
            # On-premise Ollama uses OpenAI-compatible API
            provider_meta = self.catalog.get_provider("onprem")

            # For Azure deployment - uncomment this block
            # return AzureChatOpenAI(
            #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     **kwargs
            # )

            # For local/on-prem - comment this when deploying to Azure
            return ChatOpenAI(
                base_url=provider_meta.api_base_url,
                api_key=api_key,
                model=normalized_model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        else:
            # For Azure deployment - uncomment this block
            # logger.warning(f"Unknown model type: {llm_config.model}, defaulting to Azure OpenAI")
            # return AzureChatOpenAI(
            #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     **kwargs
            # )

            # For local/standard OpenAI - comment this when deploying to Azure
            raise ValueError(f"Unsupported provider: {model_meta.provider}")

    def _get_or_create_model(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> BaseChatModel:
        """
        Get cached model or create new one

        Args:
            model_id: Model ID
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters

        Returns:
            LangChain chat model
        """
        # Create cache key
        cache_key = f"{model_id}:{temperature}:{max_tokens}"

        # Return cached model if exists
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Create new model
        model = self._create_model(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Cache it
        self._model_cache[cache_key] = model

        return model

    async def ainvoke(
        self,
        model: str,
        messages: List[BaseMessage],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AIMessage:
        """
        Invoke LLM asynchronously

        Args:
            model: Model ID
            messages: List of messages
            temperature: Temperature (0.0-2.0)
            max_tokens: Max output tokens
            system_prompt: Optional system prompt
            **kwargs: Additional model parameters

        Returns:
            AI message response

        Raises:
            ValueError: If model not found or API key missing
            Exception: If API call fails
        """
        # Prepend system message if provided
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        # Get model
        llm = self._get_or_create_model(
            model_id=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Invoke
        try:
            response = await llm.ainvoke(messages)

            # Log token usage if available
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                if usage:
                    logger.debug(
                        f"LLM invocation: model={model}, "
                        f"input_tokens={usage.get('input_tokens', 0)}, "
                        f"output_tokens={usage.get('output_tokens', 0)}"
                    )

            return response

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise

    async def astream(
        self,
        model: str,
        messages: List[BaseMessage],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream LLM response asynchronously

        Args:
            model: Model ID
            messages: List of messages
            temperature: Temperature (0.0-2.0)
            max_tokens: Max output tokens
            system_prompt: Optional system prompt
            **kwargs: Additional model parameters

        Yields:
            Response chunks
        """
        # Prepend system message if provided
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        # Get model
        llm = self._get_or_create_model(
            model_id=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Stream
        try:
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}", exc_info=True)
            raise

    async def check_availability(self, model: str) -> bool:
        """
        Check if model is available

        Makes a minimal API call to verify connectivity and credentials

        Args:
            model: Model ID

        Returns:
            True if model is available, False otherwise
        """
        try:
            # Get model metadata
            model_meta = self.catalog.get_model(model)
            if not model_meta:
                logger.warning(f"Unknown model: {model}")
                return False

            # Check API key
            api_key = self._get_api_key(model_meta.provider)
            if not api_key:
                logger.warning(f"Missing API key for {model_meta.provider}")
                return False

            # Make minimal test call
            response = await self.ainvoke(
                model=model,
                messages=[HumanMessage(content="test")],
                temperature=0.0,
                max_tokens=10
            )

            return True

        except Exception as e:
            logger.warning(f"Model {model} not available: {e}")
            return False

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for model invocation

        Args:
            model: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD

        Raises:
            ValueError: If model not found
        """
        model_meta = self.catalog.get_model(model)
        if not model_meta:
            raise ValueError(f"Unknown model: {model}")

        return model_meta.estimate_cost(input_tokens, output_tokens)

    def get_model_metadata(self, model: str) -> Optional[ModelMetadata]:
        """
        Get model metadata

        Args:
            model: Model ID

        Returns:
            Model metadata or None if not found
        """
        return self.catalog.get_model(model)

    def list_available_models(
        self,
        provider: Optional[str] = None,
        tier: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        List available models

        Args:
            provider: Filter by provider (anthropic, openai)
            tier: Filter by tier (premium, standard, fast)

        Returns:
            List of model metadata
        """
        if provider:
            models = self.catalog.get_models_by_provider(provider)
        elif tier:
            models = self.catalog.get_models_by_tier(tier)
        else:
            models = self.catalog.list_all_models()

        # Filter out models without API keys
        available_models = []
        for model_meta in models:
            api_key = self._get_api_key(model_meta.provider)
            if api_key:
                available_models.append(model_meta)

        return available_models


# ============================================================================
# GLOBAL PROVIDER INSTANCE
# ============================================================================

_global_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """
    Get global LLM provider instance

    Singleton pattern for reusing catalog and model cache

    Returns:
        LLM provider
    """
    global _global_provider

    if _global_provider is None:
        _global_provider = LLMProvider()

    return _global_provider
