"""
Agent designer service.
Generates agent definitions from natural language prompts using LLM.
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
        """Get ChatOpenAI client based on model configuration."""
        from langchain_openai import ChatOpenAI

        # Default to first available model or Ollama
        if not model_id:
            if self._llm_providers:
                model_id = list(self._llm_providers.keys())[0]
            else:
                model_id = "mistral-nemo-12b"

        model_config = self._llm_providers.get(model_id, {})
        provider = model_config.get("provider", "ollama")

        if provider == "ollama":
            base_url = model_config.get("base_url", "http://10.188.100.131:8004/v1")
            model_name = model_config.get("model_name", model_id)
            return ChatOpenAI(
                base_url=base_url,
                api_key="ollama",
                model=model_name,
                temperature=0.3
            )
        elif provider == "openai":
            api_key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env, self.api_key)
            model_name = model_config.get("model_name", model_id)
            return ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.3
            )
        elif provider == "anthropic":
            # For anthropic, we'll use ChatOpenAI with anthropic base_url
            # Or handle separately if needed
            raise NotImplementedError("Anthropic support via ChatAnthropic not implemented yet")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def design_from_prompt(
        self,
        user_prompt: str,
        default_model: str = "mistral-nemo-12b",
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
                "temperature": 0.7,
                "max_token": 2000,
                "top_p": 0.9,
                "max_iteration": 5
            },
            "input_schema": [],
            "output_schema": [],
            "constraints": {
                "max_steps": 5,
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

        system_prompt = """You are an AI agent designer. Analyze the user's request and design an agent specification.

Return a JSON response with this structure:
{
  "name": "Agent Name",
  "role": "Agent Role",
  "description": "What this agent does",
  "prompt": "System prompt for the agent to follow",
  "input_schema": ["list", "of", "input", "keys"],
  "output_schema": ["list", "of", "output", "keys"]
}

Rules:
1. Name should be concise and descriptive
2. Role should describe primary responsibility
3. Description should be 1-2 sentences
4. Prompt should be clear instructions for the agent
5. Input/output schema should reflect data flow
"""

        llm = self._get_llm_client(model_id)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object."

        # Invoke LLM
        response = llm.invoke(full_prompt)

        # Parse response
        try:
            agent_spec = json.loads(response.content)
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
