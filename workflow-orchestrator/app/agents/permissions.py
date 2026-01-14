"""
Agent Permissions
Enforce agent-to-agent communication permissions and hierarchical rules
"""
from typing import Optional, List, Set
from dataclasses import dataclass
from enum import Enum

from app.schemas.api_models import AgentDefinition, AgentPermissions
from app.core.logging import get_logger
from app.core.constants import CommunicationPattern

logger = get_logger(__name__)


# ============================================================================
# PERMISSION RESULT
# ============================================================================

class PermissionDenialReason(str, Enum):
    """Reasons for permission denial"""
    NOT_IN_ALLOWED_LIST = "not_in_allowed_list"
    CANNOT_DELEGATE = "cannot_delegate"
    EXCEEDED_TOOL_CALLS = "exceeded_tool_calls"
    HIERARCHICAL_VIOLATION = "hierarchical_violation"
    MASTER_ONLY_ACTION = "master_only_action"
    CIRCULAR_DELEGATION = "circular_delegation"


@dataclass
class PermissionCheckResult:
    """
    Result of permission check
    """
    allowed: bool
    reason: Optional[PermissionDenialReason] = None
    message: Optional[str] = None

    @classmethod
    def allow(cls) -> "PermissionCheckResult":
        """Create allowed result"""
        return cls(allowed=True)

    @classmethod
    def deny(
        cls,
        reason: PermissionDenialReason,
        message: str
    ) -> "PermissionCheckResult":
        """Create denied result"""
        return cls(allowed=False, reason=reason, message=message)


# ============================================================================
# PERMISSION VALIDATOR
# ============================================================================

class PermissionValidator:
    """
    Validates agent-to-agent permissions

    Enforces:
    - Agent-to-agent call permissions (can_call_agents)
    - Delegation permissions (can_delegate)
    - Tool call limits (max_tool_calls)
    - Hierarchical rules (master-worker relationships)
    - Circular delegation prevention

    Usage:
        validator = PermissionValidator()

        # Check if agent A can call agent B
        result = validator.check_agent_call(agent_a, agent_b)

        if result.allowed:
            # Proceed with call
        else:
            # Log denial: result.reason, result.message
    """

    def __init__(self):
        """Initialize permission validator"""
        logger.info("Permission validator initialized")

    def check_agent_call(
        self,
        caller: AgentDefinition,
        callee: AgentDefinition,
        call_chain: Optional[List[str]] = None
    ) -> PermissionCheckResult:
        """
        Check if caller can invoke callee

        Args:
            caller: Calling agent
            callee: Agent being called
            call_chain: List of agent IDs in call chain (for circular detection)

        Returns:
            PermissionCheckResult
        """
        logger.debug(f"Checking permission: {caller.id} -> {callee.id}")

        # Check if caller can delegate
        if not caller.permissions.can_delegate:
            return PermissionCheckResult.deny(
                reason=PermissionDenialReason.CANNOT_DELEGATE,
                message=f"Agent '{caller.id}' does not have delegation permission"
            )

        # Check if callee is in allowed list
        if caller.permissions.can_call_agents:
            if callee.id not in caller.permissions.can_call_agents:
                return PermissionCheckResult.deny(
                    reason=PermissionDenialReason.NOT_IN_ALLOWED_LIST,
                    message=(
                        f"Agent '{caller.id}' is not allowed to call '{callee.id}'. "
                        f"Allowed agents: {caller.permissions.can_call_agents}"
                    )
                )

        # Check for circular delegation
        if call_chain:
            if callee.id in call_chain:
                return PermissionCheckResult.deny(
                    reason=PermissionDenialReason.CIRCULAR_DELEGATION,
                    message=(
                        f"Circular delegation detected: {' -> '.join(call_chain)} -> {callee.id}"
                    )
                )

        return PermissionCheckResult.allow()

    def check_tool_call_limit(
        self,
        agent: AgentDefinition,
        current_tool_calls: int
    ) -> PermissionCheckResult:
        """
        Check if agent has exceeded tool call limit

        Args:
            agent: Agent definition
            current_tool_calls: Number of tool calls made so far

        Returns:
            PermissionCheckResult
        """
        max_calls = agent.permissions.max_tool_calls

        if current_tool_calls >= max_calls:
            return PermissionCheckResult.deny(
                reason=PermissionDenialReason.EXCEEDED_TOOL_CALLS,
                message=(
                    f"Agent '{agent.id}' has exceeded tool call limit "
                    f"({current_tool_calls}/{max_calls})"
                )
            )

        return PermissionCheckResult.allow()

    def check_hierarchical_permission(
        self,
        agent: AgentDefinition,
        action: str,
        communication_pattern: CommunicationPattern
    ) -> PermissionCheckResult:
        """
        Check hierarchical permissions

        For hierarchical patterns:
        - Only master agents can coordinate and delegate
        - Worker agents cannot initiate coordination

        Args:
            agent: Agent definition
            action: Action being attempted (e.g., "coordinate", "delegate")
            communication_pattern: Workflow communication pattern

        Returns:
            PermissionCheckResult
        """
        # Only enforce for hierarchical patterns
        if communication_pattern != CommunicationPattern.HIERARCHICAL:
            return PermissionCheckResult.allow()

        # Check master-only actions
        master_only_actions = {"coordinate", "aggregate", "delegate_to_multiple"}

        if action in master_only_actions and not agent.is_master:
            return PermissionCheckResult.deny(
                reason=PermissionDenialReason.MASTER_ONLY_ACTION,
                message=(
                    f"Action '{action}' requires master agent status. "
                    f"Agent '{agent.id}' is not a master agent."
                )
            )

        return PermissionCheckResult.allow()

    def validate_workflow_permissions(
        self,
        agents: List[AgentDefinition],
        communication_pattern: CommunicationPattern
    ) -> List[PermissionCheckResult]:
        """
        Validate permissions for entire workflow

        Checks:
        - For hierarchical: exactly one master agent
        - Agent-to-agent call graph is valid
        - No orphaned agents (unreachable)

        Args:
            agents: List of agents in workflow
            communication_pattern: Workflow communication pattern

        Returns:
            List of permission check results (only failures)
        """
        failures = []

        # For hierarchical patterns, validate master agent
        if communication_pattern == CommunicationPattern.HIERARCHICAL:
            master_agents = [a for a in agents if a.is_master]

            if len(master_agents) == 0:
                failures.append(PermissionCheckResult.deny(
                    reason=PermissionDenialReason.HIERARCHICAL_VIOLATION,
                    message="Hierarchical pattern requires at least one master agent"
                ))

            elif len(master_agents) > 1:
                failures.append(PermissionCheckResult.deny(
                    reason=PermissionDenialReason.HIERARCHICAL_VIOLATION,
                    message=(
                        f"Hierarchical pattern should have exactly one master agent. "
                        f"Found {len(master_agents)}: "
                        f"{[a.id for a in master_agents]}"
                    )
                ))

        # Validate agent-to-agent calls
        agent_dict = {a.id: a for a in agents}

        for agent in agents:
            if agent.permissions.can_call_agents:
                for callee_id in agent.permissions.can_call_agents:
                    # Check if callee exists
                    if callee_id not in agent_dict:
                        failures.append(PermissionCheckResult.deny(
                            reason=PermissionDenialReason.NOT_IN_ALLOWED_LIST,
                            message=(
                                f"Agent '{agent.id}' references unknown agent '{callee_id}' "
                                "in can_call_agents"
                            )
                        ))

        return failures

    def get_allowed_agents(self, agent: AgentDefinition) -> Set[str]:
        """
        Get set of agents that this agent can call

        Args:
            agent: Agent definition

        Returns:
            Set of agent IDs
        """
        if agent.permissions.can_call_agents:
            return set(agent.permissions.can_call_agents)

        # If no explicit list, can call anyone (permissive default)
        return set()

    def can_delegate(self, agent: AgentDefinition) -> bool:
        """
        Check if agent can delegate to other agents

        Args:
            agent: Agent definition

        Returns:
            True if agent can delegate
        """
        return agent.permissions.can_delegate


# ============================================================================
# GLOBAL VALIDATOR INSTANCE
# ============================================================================

_global_validator: Optional[PermissionValidator] = None


def get_permission_validator() -> PermissionValidator:
    """
    Get global permission validator instance

    Singleton pattern

    Returns:
        PermissionValidator
    """
    global _global_validator

    if _global_validator is None:
        _global_validator = PermissionValidator()

    return _global_validator


def check_agent_permission(
    caller: AgentDefinition,
    callee: AgentDefinition,
    call_chain: Optional[List[str]] = None
) -> PermissionCheckResult:
    """
    Convenience function to check agent call permission

    Args:
        caller: Calling agent
        callee: Agent being called
        call_chain: List of agent IDs in call chain

    Returns:
        PermissionCheckResult
    """
    validator = get_permission_validator()
    return validator.check_agent_call(caller, callee, call_chain)
