"""
Agent designer service.
Generates agent definitions from natural language prompts using LLM.

LLM Provider Configuration:
---------------------------
This module supports multiple LLM providers. Configure via .env file:
- OPTION 1: Ollama (On-Premise) - Set USE_OLLAMA=true
- OPTION 2: OpenRouter (Current) - Set USE_OPENROUTER=true
- OPTION 3: Azure OpenAI - Set USE_AZURE=true
- OPTION 4: OpenAI Direct - Set USE_OPENAI=true

See .env file for detailed configuration options.
"""
import json
import os
from typing import Dict, Any, Optional
from echolib.utils import new_id
from datetime import datetime


class AgentDesigner:
    """
    Agent designer service.
    Uses LLM to generate agent definitions from natural language prompts.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize designer.

        Args:
            api_key: OpenAI API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._llm_client = None
        self._llm_providers = self._load_llm_providers()

    def _load_llm_providers(self) -> Dict[str, Any]:
        """Load LLM provider configurations from llm_provider.json."""
        provider_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "llm_provider.json"
        )
        if os.path.exists(provider_file):
            with open(provider_file, 'r') as f:
                data = json.load(f)
                return {model["id"]: model for model in data.get("models", [])}
        return {}

    def _get_llm_client(self, model_id: str = None):
        """
        Get LLM client based on environment configuration.

        Provider Priority (based on .env settings):
        1. USE_AZURE=true -> Azure OpenAI
        2. USE_OPENROUTER=true -> OpenRouter (current default)
        3. USE_OLLAMA=true -> Ollama (on-premise)
        4. USE_OPENAI=true -> OpenAI Direct
        """
        from langchain_openai import ChatOpenAI
        # For Azure deployment - uncomment the line below
        # from langchain_openai import AzureChatOpenAI

        # =================================================================
        # OPTION 1: AZURE OPENAI (For Azure Deployment)
        # =================================================================
        # Uncomment this block when deploying to Azure
        # -----------------------------------------------------------------
        # if os.getenv("USE_AZURE", "false").lower() == "true":
        #     return AzureChatOpenAI(
        #         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        #         api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #         temperature=0.3,
        #         max_tokens=4000
        #     )

        # =================================================================
        # OPTION 2: OPENROUTER (Current - For Development)
        # =================================================================
        # Using OpenRouter with free tier model - CURRENTLY ACTIVE
        # -----------------------------------------------------------------
        if os.getenv("USE_OPENROUTER", "true").lower() == "true":
            openrouter_key = os.getenv(
                "OPENROUTER_API_KEY",
                "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"
            )
            openrouter_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            openrouter_model = os.getenv("OPENROUTER_MODEL", "mistralai/devstral-2512:free")
            return ChatOpenAI(
                base_url=openrouter_url,
                api_key=openrouter_key,
                model=openrouter_model,
                temperature=0.3,
                max_tokens=4000
            )

        # =================================================================
        # OPTION 3: OLLAMA (On-Premise/Local)
        # =================================================================
        # Uncomment this block for local Ollama deployment
        # -----------------------------------------------------------------
        # if os.getenv("USE_OLLAMA", "false").lower() == "true":
        #     ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.188.100.131:8004/v1")
        #     ollama_model = os.getenv("OLLAMA_MODEL", "mistral-nemo:12b-instruct-2407-fp16")
        #     return ChatOpenAI(
        #         base_url=ollama_url,
        #         api_key="ollama",
        #         model=ollama_model,
        #         temperature=0.3,
        #         max_tokens=4000
        #     )

        # =================================================================
        # OPTION 4: OPENAI DIRECT
        # =================================================================
        # Uncomment this block for direct OpenAI API
        # -----------------------------------------------------------------
        # if os.getenv("USE_OPENAI", "false").lower() == "true":
        #     openai_key = os.getenv("OPENAI_API_KEY")
        #     openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        #     return ChatOpenAI(
        #         api_key=openai_key,
        #         model=openai_model,
        #         temperature=0.3,
        #         max_tokens=4000
        #     )

        # =================================================================
        # FALLBACK: Use model from llm_provider.json
        # =================================================================
        if not model_id:
            # Find default model
            for mid, config in self._llm_providers.items():
                if config.get("is_default", False):
                    model_id = mid
                    break
            if not model_id and self._llm_providers:
                model_id = list(self._llm_providers.keys())[0]

        if model_id and model_id in self._llm_providers:
            model_config = self._llm_providers[model_id]
            provider = model_config.get("provider", "openrouter")

            if provider == "openrouter":
                api_key = os.getenv(
                    model_config.get("api_key_env", "OPENROUTER_API_KEY"),
                    "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"
                )
                return ChatOpenAI(
                    base_url=model_config.get("base_url", "https://openrouter.ai/api/v1"),
                    api_key=api_key,
                    model=model_config.get("model_name"),
                    temperature=0.3,
                    max_tokens=4000
                )
            elif provider == "ollama":
                return ChatOpenAI(
                    base_url=model_config.get("base_url", "http://localhost:11434/v1"),
                    api_key="ollama",
                    model=model_config.get("model_name"),
                    temperature=0.3,
                    max_tokens=4000
                )
            elif provider == "openai":
                api_key = os.getenv(model_config.get("api_key_env", "OPENAI_API_KEY"), self.api_key)
                return ChatOpenAI(
                    api_key=api_key,
                    model=model_config.get("model_name", "gpt-4o-mini"),
                    temperature=0.3,
                    max_tokens=4000
                )

        # Ultimate fallback: OpenRouter
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
            model="mistralai/devstral-2512:free",
            temperature=0.3,
            max_tokens=4000
        )

    def design_from_prompt(
        self,
        user_prompt: str,
        default_model: str = "openrouter-devstral",
        icon: str = "🤖",
        tools: list = None,
        variables: list = None
    ) -> Dict[str, Any]:
        """
        Design agent from user prompt using LLM.

        Args:
            user_prompt: Natural language description of agent
            default_model: Model ID to use for agent
            icon: Emoji icon for agent
            tools: List of tool names
            variables: List of variable definitions

        Returns:
            Agent definition dict
        """
        if tools is None:
            tools = []
        if variables is None:
            variables = []

        # Use LLM to analyze prompt
        try:
            agent_spec = self._design_with_llm(user_prompt, default_model)
        except Exception as e:
            print(f"LLM design failed, using basic structure: {e}")
            agent_spec = self._design_basic(user_prompt)

        # Build agent definition
        agent_id = new_id("agt_")
        timestamp = datetime.utcnow().isoformat()

        # Get LLM-suggested settings or use defaults
        llm_settings = agent_spec.get("settings", {})
        temperature = llm_settings.get("temperature", 0.7)
        max_tokens = llm_settings.get("max_tokens", 2000)
        top_p = llm_settings.get("top_p", 0.9)
        max_iterations = llm_settings.get("max_iterations", 5)

        agent = {
            "agent_id": agent_id,
            "name": agent_spec.get("name", "Agent"),
            "icon": icon,
            "role": agent_spec.get("role", "Processing"),
            "description": agent_spec.get("description", user_prompt[:200]),
            "prompt": agent_spec.get("prompt", user_prompt),
            "model": default_model,
            "tools": tools,
            "variables": variables,
            "settings": {
                "temperature": temperature,
                "max_token": max_tokens,
                "top_p": top_p,
                "max_iteration": max_iterations
            },
            "input_schema": agent_spec.get("input_schema", []),
            "output_schema": agent_spec.get("output_schema", []),
            "constraints": {
                "max_steps": max_iterations,
                "timeout_seconds": 60
            },
            "permissions": {
                "can_call_agents": False,
                "allowed_agents": []
            },
            "metadata": {
                "created_by": "agent_designer",
                "created_at": timestamp,
                "tags": ["auto-generated"]
            }
        }

        return agent

    def _design_with_llm(
        self,
        user_prompt: str,
        model_id: str
    ) -> Dict[str, Any]:
        """Design agent using LLM analysis."""

        system_prompt = """You are an AI agent designer. Analyze the user's request and design a complete agent specification.

Return a JSON response with this exact structure:
{
  "name": "Creative and Memorable Agent Name",
  "role": "Professional Role/Title",
  "description": "What this agent does (1-2 sentences)",
  "prompt": "Detailed system prompt/instructions for the agent",
  "input_schema": ["list", "of", "input", "keys"],
  "output_schema": ["list", "of", "output", "keys"],
  "settings": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "max_iterations": 5
  }
}

IMPORTANT RULES:

1. NAME: Must be creative, memorable, and professional. Examples:
   - For code review: "CodeCraft Pro", "PyReviewer Elite", "SyntaxMaster"
   - For content writing: "ContentForge", "WordSmith Pro", "NarrativeGenius"
   - For data analysis: "DataWiz", "InsightEngine", "AnalyticsPro"
   - NEVER use generic names like "Custom Agent", "AI Agent", or "New Agent"

2. ROLE: A professional job title (e.g., "Senior Python Developer", "Content Strategist")

3. DESCRIPTION: Clear 1-2 sentence explanation of what the agent does

4. PROMPT: Detailed instructions for the agent to follow when executing tasks

5. SETTINGS: Tune based on task type:
   - temperature: 0.1-0.3 for factual/precise tasks (code, math), 0.5-0.7 for balanced tasks, 0.8-1.0 for creative tasks
   - max_tokens: 1000-2000 for short outputs, 2000-4000 for detailed responses
   - top_p: 0.9 default, lower (0.5-0.7) for more focused outputs
   - max_iterations: 3-5 for simple tasks, 5-10 for complex multi-step tasks
"""

        llm = self._get_llm_client(model_id)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object."

        # Invoke LLM
        response = llm.invoke(full_prompt)

        # Parse response
        try:
            # Handle response content - may be string or have content attribute
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to extract JSON from response
            # Sometimes LLM wraps JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            agent_spec = json.loads(content)
            return agent_spec
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return self._design_basic(user_prompt)

    def _design_basic(self, user_prompt: str) -> Dict[str, Any]:
        """Basic agent structure without LLM."""
        return {
            "name": "Custom Agent",
            "role": "Processing",
            "description": user_prompt[:200],
            "prompt": user_prompt,
            "input_schema": [],
            "output_schema": []
        }
