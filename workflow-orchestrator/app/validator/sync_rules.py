"""
Synchronous Validation Rules
Schema, topology, IO, and reference validation (no external calls)
"""
from typing import List, Set, Dict
import json
from pathlib import Path

from app.schemas.api_models import AgentSystemDesign
from app.validator.errors import (
    ValidationError,
    schema_error,
    topology_error,
    reference_error,
    configuration_warning,
    info_message
)
from app.core.constants import SystemLimits, CommunicationPattern
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# AGENT UNIQUENESS
# ============================================================================

def validate_agent_uniqueness(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Ensure all agent IDs are unique

    CRITICAL: Duplicate IDs will cause runtime failures
    """
    errors = []
    agent_ids = [agent.id for agent in system.agents]

    seen = set()
    for agent_id in agent_ids:
        if agent_id in seen:
            errors.append(schema_error(
                location=f"agents.{agent_id}",
                message=f"Duplicate agent ID: '{agent_id}'",
                suggestion="Each agent must have a unique ID. Rename one of the agents."
            ))
        seen.add(agent_id)

    return errors


# ============================================================================
# TOOL REFERENCES
# ============================================================================

def validate_tool_references(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Check if tools referenced by agents are defined inline.

    NOTE: This is now a WARNING, not an error, because:
    - MCP-first architecture: tools may be registered in MCP registry
    - Runtime will bind tools dynamically from MCP server
    - Inline tool definitions are optional metadata/hints
    """
    warnings = []

    # Get all defined tool names (inline definitions)
    defined_tools: Set[str] = {tool.name for tool in system.tools}

    # Check each agent's tool references
    for agent in system.agents:
        for tool_name in agent.tools:
            if tool_name not in defined_tools:
                # Changed to warning - tools may be MCP-registered
                warnings.append(configuration_warning(
                    location=f"agents.{agent.id}.tools",
                    message=f"Agent '{agent.id}' references tool '{tool_name}' (not defined inline - ensure it's registered in MCP)",
                    suggestion=f"Tool '{tool_name}' will be resolved from MCP registry at runtime"
                ))

    return warnings


# ============================================================================
# WORKFLOW REFERENCES
# ============================================================================

def validate_workflow_references(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Ensure all workflow steps reference valid agents

    CRITICAL: Invalid agent references will cause runtime failures
    """
    errors = []

    # Get all agent IDs
    valid_agent_ids: Set[str] = {agent.id for agent in system.agents}

    # Check each workflow
    for workflow in system.workflows:
        for step_idx, step in enumerate(workflow.steps):
            if step.agent_id not in valid_agent_ids:
                errors.append(reference_error(
                    location=f"workflows.{workflow.name}.steps[{step_idx}]",
                    message=f"Step references undefined agent: '{step.agent_id}'",
                    missing_ref=step.agent_id,
                    suggestion=f"Ensure agent '{step.agent_id}' is defined in agents section"
                ))

    return errors


# ============================================================================
# TOPOLOGY VALIDATION
# ============================================================================

def validate_topology(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate graph topology (no cycles in sequential/hierarchical)

    CRITICAL for sequential and hierarchical workflows

    NOTE: For hierarchical SYSTEM patterns, we allow the master agent to appear
    multiple times (at start and end) as this is a legitimate orchestration pattern
    where the master coordinates at beginning and synthesizes at end.

    NOTE: For parallel SYSTEM patterns, fan-out/fan-in is allowed and should NOT
    be flagged as a cycle. Skip strict cycle checking for parallel systems.

    NOTE: For sequential workflows with a coordinator agent (appears at start and end),
    we allow this orchestration pattern without flagging it as a cycle.
    """
    errors = []

    # Skip cycle checking for PARALLEL system patterns
    # Parallel workflows naturally have fan-out/fan-in which is not a cycle
    if system.communication_pattern == CommunicationPattern.PARALLEL:
        return errors

    # Get master agent ID for hierarchical systems
    master_agent_id = None
    if system.communication_pattern == CommunicationPattern.HIERARCHICAL:
        for agent in system.agents:
            if agent.is_master:
                master_agent_id = agent.id
                break

    for workflow in system.workflows:
        # Skip cycle checking for PARALLEL workflows (at workflow level)
        # This handles cases where system pattern differs from workflow pattern
        if workflow.communication_pattern == CommunicationPattern.PARALLEL:
            continue

        if workflow.communication_pattern in [CommunicationPattern.SEQUENTIAL, CommunicationPattern.HIERARCHICAL]:
            # Build adjacency list
            graph: Dict[str, Set[str]] = {}

            for step in workflow.steps:
                if step.agent_id not in graph:
                    graph[step.agent_id] = set()

            # Detect coordinator pattern: same agent at first and last position
            # This is a legitimate orchestration pattern (coordinate → work → aggregate)
            coordinator_agent_id = None
            if len(workflow.steps) >= 2:
                first_agent = workflow.steps[0].agent_id
                last_agent = workflow.steps[-1].agent_id
                if first_agent == last_agent:
                    coordinator_agent_id = first_agent
                    logger.info(f"Detected coordinator pattern in workflow '{workflow.name}': agent '{coordinator_agent_id}' at start and end")

            # Also detect coordinator by common naming patterns
            if coordinator_agent_id is None:
                coordinator_keywords = ["coordinator", "orchestrator", "master", "manager", "supervisor", "planner"]
                for step in workflow.steps:
                    agent_id_lower = step.agent_id.lower()
                    if any(keyword in agent_id_lower for keyword in coordinator_keywords):
                        # Check if this agent appears more than once in the workflow
                        appearances = sum(1 for s in workflow.steps if s.agent_id == step.agent_id)
                        if appearances > 1:
                            coordinator_agent_id = step.agent_id
                            logger.info(f"Detected coordinator agent by naming pattern in workflow '{workflow.name}': '{coordinator_agent_id}'")
                            break

            # For sequential, each step connects to the next
            if workflow.communication_pattern == CommunicationPattern.SEQUENTIAL:
                for i in range(len(workflow.steps) - 1):
                    current = workflow.steps[i].agent_id
                    next_agent = workflow.steps[i + 1].agent_id

                    # For hierarchical SYSTEM patterns, skip edges TO the master agent
                    # This allows master to appear at start and end (orchestration pattern)
                    if system.communication_pattern == CommunicationPattern.HIERARCHICAL:
                        if next_agent == master_agent_id:
                            # This is the master agent receiving control back - allowed
                            continue

                    # For coordinator pattern in sequential workflows, skip edge back to coordinator
                    # This allows coordinator to appear at start and end without creating a cycle
                    if coordinator_agent_id is not None and next_agent == coordinator_agent_id:
                        # This is the coordinator receiving control back - allowed
                        continue

                    graph[current].add(next_agent)

            # Check for cycles
            if _has_cycle(graph):
                errors.append(topology_error(
                    location=f"workflows.{workflow.name}",
                    message=f"Workflow has a cycle, incompatible with {workflow.communication_pattern} pattern",
                    suggestion="Remove circular dependencies or use 'graph' pattern instead"
                ))

    return errors


def _has_cycle(graph: Dict[str, Set[str]]) -> bool:
    """
    Detect cycles in directed graph using DFS

    Returns True if cycle detected
    """
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True

    return False


# ============================================================================
# COMMUNICATION PATTERNS
# ============================================================================

def validate_communication_patterns(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate communication pattern consistency

    Warnings for pattern mismatches
    """
    warnings = []

    # Check if workflow patterns match system pattern
    for workflow in system.workflows:
        if workflow.communication_pattern != system.communication_pattern:
            warnings.append(configuration_warning(
                location=f"workflows.{workflow.name}.communication_pattern",
                message=f"Workflow pattern '{workflow.communication_pattern}' differs from system pattern '{system.communication_pattern}'",
                suggestion="Consider aligning workflow and system communication patterns for consistency"
            ))

    # Validate parallel workflows have parallel_with properly set
    for workflow in system.workflows:
        if workflow.communication_pattern == CommunicationPattern.PARALLEL:
            has_parallel_steps = any(
                step.parallel_with for step in workflow.steps
            )
            if not has_parallel_steps and len(workflow.steps) > 1:
                warnings.append(configuration_warning(
                    location=f"workflows.{workflow.name}",
                    message="Parallel workflow has no steps marked for parallel execution",
                    suggestion="Use 'parallel_with' in steps to indicate concurrent execution"
                ))

    return warnings


# ============================================================================
# HIERARCHICAL STRUCTURE
# ============================================================================

def validate_hierarchical(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate hierarchical structure requirements

    CRITICAL for hierarchical pattern
    """
    errors = []

    # Only validate if system uses hierarchical pattern
    if system.communication_pattern != CommunicationPattern.HIERARCHICAL:
        return errors

    # Count master agents
    master_agents = [agent for agent in system.agents if agent.is_master]

    if len(master_agents) == 0:
        errors.append(schema_error(
            location="agents",
            message="Hierarchical pattern requires at least one master agent",
            suggestion="Set is_master=true for the coordinator agent"
        ))
    elif len(master_agents) > 1:
        errors.append(schema_error(
            location="agents",
            message=f"Multiple master agents found ({len(master_agents)}). Hierarchical pattern should have exactly one.",
            suggestion="Mark only one agent as master (coordinator)"
        ))

    return errors


# ============================================================================
# LLM CONFIGURATIONS
# ============================================================================

def validate_llm_configs(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate LLM configurations are reasonable

    Warnings for suboptimal configurations
    """
    warnings = []

    for agent in system.agents:
        llm = agent.llm_config

        # Check temperature ranges
        if llm.temperature > 1.5:
            warnings.append(configuration_warning(
                location=f"agents.{agent.id}.llm_config.temperature",
                message=f"High temperature ({llm.temperature}) may produce inconsistent results",
                suggestion="Consider using temperature between 0.3-0.8 for production tasks"
            ))

        # Check max_tokens for complex tasks
        if llm.max_tokens < 500 and "analysis" in agent.role.lower():
            warnings.append(configuration_warning(
                location=f"agents.{agent.id}.llm_config.max_tokens",
                message=f"Low max_tokens ({llm.max_tokens}) for analysis task",
                suggestion="Consider increasing max_tokens to at least 1000 for analysis tasks"
            ))

    return warnings


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

def validate_system_prompts(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate system prompts are meaningful

    Warnings for short or unclear prompts
    """
    warnings = []

    for agent in system.agents:
        prompt = agent.system_prompt

        # Check if prompt is too short
        if len(prompt) < SystemLimits.MIN_SYSTEM_PROMPT_LENGTH:
            warnings.append(configuration_warning(
                location=f"agents.{agent.id}.system_prompt",
                message="System prompt is very short",
                suggestion="Consider providing more context and instructions for better agent performance"
            ))

        # Check if prompt mentions the agent's role
        if agent.role.lower() not in prompt.lower():
            warnings.append(info_message(
                location=f"agents.{agent.id}.system_prompt",
                message="System prompt doesn't mention agent's role",
                suggestion=f"Consider incorporating '{agent.role}' into the system prompt for clarity"
            ))

    return warnings


# ============================================================================
# SYSTEM LIMITS
# ============================================================================

def validate_system_limits(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate system stays within limits

    CRITICAL: Exceeding limits may cause performance issues
    """
    errors = []

    # Check agent count
    if len(system.agents) > SystemLimits.MAX_AGENTS_PER_WORKFLOW:
        errors.append(schema_error(
            location="agents",
            message=f"Too many agents: {len(system.agents)} (max: {SystemLimits.MAX_AGENTS_PER_WORKFLOW})",
            suggestion="Consider breaking into multiple workflows or simplifying the system"
        ))

    # Check tools per agent
    for agent in system.agents:
        if len(agent.tools) > SystemLimits.MAX_TOOLS_PER_AGENT:
            errors.append(schema_error(
                location=f"agents.{agent.id}.tools",
                message=f"Too many tools: {len(agent.tools)} (max: {SystemLimits.MAX_TOOLS_PER_AGENT})",
                suggestion="Reduce the number of tools or split agent responsibilities"
            ))

    # Check workflow steps
    for workflow in system.workflows:
        if len(workflow.steps) > SystemLimits.MAX_WORKFLOW_STEPS:
            errors.append(schema_error(
                location=f"workflows.{workflow.name}.steps",
                message=f"Too many steps: {len(workflow.steps)} (max: {SystemLimits.MAX_WORKFLOW_STEPS})",
                suggestion="Break into smaller sub-workflows"
            ))

    return errors


# ============================================================================
# MAIN SYNC VALIDATION
# ============================================================================

def run_all_sync_validations(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Run all synchronous validation rules

    This is the main entry point for sync validation
    """
    logger.info(f"Running synchronous validation for: {system.system_name}")

    all_errors: List[ValidationError] = []

    # Critical validations (must pass)
    uniqueness_errors = validate_agent_uniqueness(system)
    if uniqueness_errors:
        for e in uniqueness_errors:
            logger.error(f"[UNIQUENESS] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(uniqueness_errors)

    tool_errors = validate_tool_references(system)
    if tool_errors:
        for e in tool_errors:
            logger.warning(f"[TOOL_REF] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(tool_errors)

    workflow_errors = validate_workflow_references(system)
    if workflow_errors:
        for e in workflow_errors:
            logger.error(f"[WORKFLOW_REF] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(workflow_errors)

    topology_errors = validate_topology(system)
    if topology_errors:
        for e in topology_errors:
            logger.error(f"[TOPOLOGY] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(topology_errors)

    hierarchical_errors = validate_hierarchical(system)
    if hierarchical_errors:
        for e in hierarchical_errors:
            logger.error(f"[HIERARCHICAL] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(hierarchical_errors)

    limit_errors = validate_system_limits(system)
    if limit_errors:
        for e in limit_errors:
            logger.error(f"[LIMITS] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(limit_errors)

    # Advisory validations (warnings/info)
    pattern_warnings = validate_communication_patterns(system)
    if pattern_warnings:
        for e in pattern_warnings:
            logger.info(f"[PATTERNS] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(pattern_warnings)

    llm_warnings = validate_llm_configs(system)
    if llm_warnings:
        for e in llm_warnings:
            logger.info(f"[LLM_CONFIG] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(llm_warnings)

    prompt_warnings = validate_system_prompts(system)
    if prompt_warnings:
        for e in prompt_warnings:
            logger.info(f"[PROMPTS] {e.severity.value}: {e.message} @ {e.location}")
    all_errors.extend(prompt_warnings)

    # Summary logging
    error_count = sum(1 for e in all_errors if e.severity.value == "error")
    warning_count = sum(1 for e in all_errors if e.severity.value == "warning")
    info_count = sum(1 for e in all_errors if e.severity.value == "info")
    logger.info(f"Synchronous validation complete: {len(all_errors)} issues found (errors={error_count}, warnings={warning_count}, info={info_count})")

    return all_errors
