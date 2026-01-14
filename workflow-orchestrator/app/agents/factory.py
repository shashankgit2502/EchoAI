"""
Agent Factory
Create runtime agent instances with LLM integration
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool

from app.schemas.api_models import AgentDefinition
from app.services.llm_provider import get_llm_provider
from app.agents.registry import get_agent_registry
from app.core.logging import get_logger
from app.core.telemetry import trace_operation
from app.core.constants import TelemetrySpan

logger = get_logger(__name__)


# ============================================================================
# RUNTIME AGENT
# ============================================================================

@dataclass
class RuntimeAgent:
    """
    Runtime agent instance with LLM and tools

    Wraps LangChain agent with additional metadata
    """
    agent_id: str
    definition: AgentDefinition
    agent: Any  # LangChain agent instance
    llm_model: str
    tools: List[Tool]

    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke agent asynchronously

        Args:
            input_data: Input data for agent (should contain "messages" key)

        Returns:
            Agent response
        """
        with trace_operation(
            TelemetrySpan.AGENT_EXECUTION,
            {
                "agent_id": self.agent_id,
                "llm_model": self.llm_model,
                "tool_count": len(self.tools)
            }
        ):
            logger.debug(f"Invoking agent: {self.agent_id}")

            try:
                response = await self.agent.ainvoke(input_data)
                logger.debug(f"Agent {self.agent_id} completed successfully")
                return response

            except Exception as e:
                logger.error(f"Agent {self.agent_id} execution failed: {e}", exc_info=True)
                raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "role": self.definition.role,
            "llm_model": self.llm_model,
            "tool_count": len(self.tools),
            "is_master": self.definition.is_master
        }


# ============================================================================
# AGENT FACTORY
# ============================================================================

class AgentFactory:
    """
    Factory for creating runtime agent instances

    Creates LangChain agents with:
    - LLM integration via LLM Provider
    - Tool binding
    - System prompts
    - Permissions enforcement

    Usage:
        factory = AgentFactory()

        # Create agent from definition
        agent = await factory.create_agent(agent_def, tools)

        # Create agent by ID
        agent = await factory.create_agent_by_id("security_analyst", tools)

        # Invoke agent
        response = await agent.ainvoke({"input": "Analyze this alert"})
    """

    def __init__(self):
        """Initialize agent factory"""
        self.llm_provider = get_llm_provider()
        self.agent_registry = get_agent_registry()
        self._cache: Dict[str, RuntimeAgent] = {}

        logger.info("Agent factory initialized")

    def _build_system_prompt(self, agent_def: AgentDefinition) -> str:
        """
        Build system prompt for agent

        Args:
            agent_def: Agent definition

        Returns:
            System prompt string
        """
        # System prompt with role and responsibilities
        system_prompt = f"""{agent_def.system_prompt}

Role: {agent_def.role}

Responsibilities:
{chr(10).join(f"- {resp}" for resp in agent_def.responsibilities)}

You have access to tools to help you complete your tasks.
Use tools when necessary and provide clear reasoning in your responses.
"""
        return system_prompt

    async def create_agent(
        self,
        agent_def: AgentDefinition,
        tools: Optional[List[Tool]] = None,
        use_cache: bool = True
    ) -> RuntimeAgent:
        """
        Create runtime agent from definition

        Args:
            agent_def: Agent definition
            tools: List of tools (empty list if None)
            use_cache: Use cached agent if available

        Returns:
            RuntimeAgent

        Raises:
            ValueError: If LLM model not available
        """
        # Check cache
        if use_cache and agent_def.id in self._cache:
            logger.debug(f"Using cached agent: {agent_def.id}")
            return self._cache[agent_def.id]

        with trace_operation(
            TelemetrySpan.AGENT_CREATION,
            {"agent_id": agent_def.id, "llm_model": agent_def.llm_config.model}
        ):
            logger.info(f"Creating agent: {agent_def.id}")

            # Default to empty tools list
            tools = tools or []

            # Filter tools based on agent's tool list
            if agent_def.tools:
                tool_names = set(agent_def.tools)
                filtered_tools = [t for t in tools if t.name in tool_names]
                logger.debug(
                    f"Agent {agent_def.id} using {len(filtered_tools)}/{len(tools)} tools"
                )
            else:
                filtered_tools = tools

            # Get LLM model identifier
            # LangChain v1 create_agent accepts model string directly
            model_id = agent_def.llm_config.model

            # Verify model is available
            try:
                # Just check if model exists
                _ = self.llm_provider._get_or_create_model(
                    model_id=model_id,
                    temperature=agent_def.llm_config.temperature,
                    max_tokens=agent_def.llm_config.max_tokens
                )
            except Exception as e:
                logger.error(f"Failed to verify LLM for agent {agent_def.id}: {e}")
                raise ValueError(
                    f"Cannot create agent {agent_def.id}: LLM model "
                    f"{model_id} not available"
                )

            # Build system prompt
            system_prompt = self._build_system_prompt(agent_def)

            # Create agent using LangChain v1 API
            agent = create_agent(
                model=model_id,
                tools=filtered_tools,
                system_prompt=system_prompt
            )

            # Create runtime agent
            runtime_agent = RuntimeAgent(
                agent_id=agent_def.id,
                definition=agent_def,
                agent=agent,
                llm_model=model_id,
                tools=filtered_tools
            )

            # Cache it
            if use_cache:
                self._cache[agent_def.id] = runtime_agent

            logger.info(
                f"Created agent: {agent_def.id} with {len(filtered_tools)} tools"
            )

            return runtime_agent

    async def create_agent_by_id(
        self,
        agent_id: str,
        tools: Optional[List[Tool]] = None,
        use_cache: bool = True
    ) -> Optional[RuntimeAgent]:
        """
        Create agent by ID (loads from registry)

        Args:
            agent_id: Agent ID
            tools: List of tools
            use_cache: Use cached agent if available

        Returns:
            RuntimeAgent or None if not found
        """
        # Load agent definition
        agent_def = self.agent_registry.get_agent(agent_id)

        if not agent_def:
            logger.warning(f"Agent not found: {agent_id}")
            return None

        # Create agent
        return await self.create_agent(agent_def, tools, use_cache)

    async def create_agents_batch(
        self,
        agent_ids: List[str],
        tools: Optional[List[Tool]] = None
    ) -> Dict[str, RuntimeAgent]:
        """
        Create multiple agents in batch

        Args:
            agent_ids: List of agent IDs
            tools: List of tools

        Returns:
            Dictionary of agent_id -> RuntimeAgent
        """
        logger.info(f"Creating {len(agent_ids)} agents in batch")

        agents = {}

        for agent_id in agent_ids:
            try:
                agent = await self.create_agent_by_id(agent_id, tools)
                if agent:
                    agents[agent_id] = agent
            except Exception as e:
                logger.error(f"Failed to create agent {agent_id}: {e}")

        logger.info(f"Created {len(agents)}/{len(agent_ids)} agents successfully")

        return agents

    def clear_cache(self):
        """Clear agent cache"""
        logger.info("Clearing agent cache")
        self._cache.clear()

    def get_cached_agent(self, agent_id: str) -> Optional[RuntimeAgent]:
        """
        Get cached agent

        Args:
            agent_id: Agent ID

        Returns:
            RuntimeAgent or None if not cached
        """
        return self._cache.get(agent_id)

    def list_cached_agents(self) -> List[str]:
        """
        List cached agent IDs

        Returns:
            List of agent IDs
        """
        return list(self._cache.keys())


# ============================================================================
# GLOBAL FACTORY INSTANCE
# ============================================================================

_global_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """
    Get global agent factory instance

    Singleton pattern for reusing cached agents

    Returns:
        AgentFactory
    """
    global _global_factory

    if _global_factory is None:
        _global_factory = AgentFactory()

    return _global_factory


def create_runtime_agent(
    agent_def: AgentDefinition,
    tools: Optional[List[Tool]] = None
) -> RuntimeAgent:
    """
    Convenience function to create agent

    Args:
        agent_def: Agent definition
        tools: List of tools

    Returns:
        RuntimeAgent
    """
    factory = get_agent_factory()
    return factory.create_agent(agent_def, tools)
