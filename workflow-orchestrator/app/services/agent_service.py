"""
Agent Service
Service layer wrapper for agent operations
Provides clean interface for internal API communication
"""
from typing import Optional, List
from app.schemas.api_models import (
    CreateAgentRequest,
    CreateAgentResponse,
    ValidateAgentPermissionsRequest,
    ValidateAgentPermissionsResponse,
    ListAgentsResponse,
    AgentDefinition
)
from app.agents.registry import AgentRegistry
from app.agents.factory import AgentFactory
from app.agents.permissions import PermissionValidator
from app.core.logging import get_logger

logger = get_logger(__name__)


class AgentService:
    """
    Service layer for agent operations

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Async + idempotent
    - Ready for microservice extraction

    Responsibilities:
    - Agent registry operations (load/save)
    - Agent factory (runtime instantiation)
    - Permission validation
    """

    def __init__(self):
        """Initialize agent service"""
        self._registry = AgentRegistry()
        self._factory = AgentFactory()
        self._permission_validator = PermissionValidator()
        logger.info("AgentService initialized")

    async def create_runtime_agent(
        self,
        request: CreateAgentRequest
    ) -> CreateAgentResponse:
        """
        Create runtime agent instance

        Args:
            request: Agent creation request

        Returns:
            CreateAgentResponse with creation status
        """
        try:
            logger.info(f"Creating runtime agent: {request.agent_definition.id}")

            # Save to registry
            self._registry.save_agent(request.agent_definition)

            # Create runtime instance
            agent = self._factory.create_agent(request.agent_definition)

            logger.info(f"Agent created: {request.agent_definition.id}")

            return CreateAgentResponse(
                agent_id=request.agent_definition.id,
                created=True,
                error=None
            )

        except Exception as e:
            logger.error(f"Agent creation failed: {e}", exc_info=True)
            return CreateAgentResponse(
                agent_id=request.agent_definition.id,
                created=False,
                error=str(e)
            )

    def validate_permissions(
        self,
        request: ValidateAgentPermissionsRequest
    ) -> ValidateAgentPermissionsResponse:
        """
        Validate agent-to-agent permissions

        Args:
            request: Permission validation request

        Returns:
            ValidateAgentPermissionsResponse with allowed status
        """
        logger.info(f"Validating permission: {request.agent_id} → {request.target_agent_id}")

        # Load agents
        agent = self._registry.load_agent(request.agent_id)
        target = self._registry.load_agent(request.target_agent_id)

        if not agent or not target:
            return ValidateAgentPermissionsResponse(
                allowed=False,
                reason="Agent not found"
            )

        # Check permissions
        result = self._permission_validator.check_agent_call(
            caller=agent,
            callee=target
        )

        return ValidateAgentPermissionsResponse(
            allowed=result.allowed,
            reason=result.message if not result.allowed else None
        )

    def list_agents(self) -> ListAgentsResponse:
        """
        List all registered agents

        Returns:
            ListAgentsResponse with all agents
        """
        logger.info("Listing all agents")

        agents = self._registry.list_agents()

        return ListAgentsResponse(
            agents=agents,
            count=len(agents)
        )

    def get_agent(self, agent_id: str) -> Optional[AgentDefinition]:
        """
        Get single agent by ID

        Args:
            agent_id: Agent identifier

        Returns:
            AgentDefinition or None
        """
        logger.info(f"Getting agent: {agent_id}")
        return self._registry.load_agent(agent_id)

    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete agent from registry

        Args:
            agent_id: Agent identifier

        Returns:
            True if deleted, False otherwise
        """
        logger.info(f"Deleting agent: {agent_id}")
        return self._registry.delete_agent(agent_id)


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """
    Get singleton agent service instance

    Returns:
        AgentService instance
    """
    global _agent_service

    if _agent_service is None:
        _agent_service = AgentService()

    return _agent_service
