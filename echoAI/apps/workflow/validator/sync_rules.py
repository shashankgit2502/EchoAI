"""
Synchronous validation rules.
Fast, deterministic checks for workflow correctness.
"""
import jsonschema
from typing import Dict, Any
from echolib.schemas import WORKFLOW_SCHEMA, AGENT_SCHEMA
from .errors import ValidationResult


def validate_workflow_schema(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate workflow matches JSON schema."""
    try:
        jsonschema.validate(instance=workflow, schema=WORKFLOW_SCHEMA)
    except jsonschema.ValidationError as e:
        result.add_error(f"Workflow schema validation failed: {e.message}")


def validate_agent_schemas(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate all agents match agent schema."""
    for agent_id in workflow.get("agents", []):
        if agent_id not in agent_registry:
            continue  # Will be caught by validate_agents_exist

        agent = agent_registry[agent_id]
        try:
            jsonschema.validate(instance=agent, schema=AGENT_SCHEMA)
        except jsonschema.ValidationError as e:
            result.add_error(f"Agent '{agent_id}' schema invalid: {e.message}")


def validate_agents_exist(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Check all referenced agents exist."""
    for agent_id in workflow.get("agents", []):
        if agent_id not in agent_registry:
            result.add_error(f"Agent '{agent_id}' not found")


def validate_tools(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    tool_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate all agent tools exist in tool registry."""
    # Builtin tool types that don't require registry lookup
    BUILTIN_TOOL_PREFIXES = ("builtin_code", "builtin_subworkflow", "builtin_mcp_server")

    for agent_id in workflow.get("agents", []):
        if agent_id not in agent_registry:
            continue

        agent = agent_registry[agent_id]
        for tool_id in agent.get("tools", []):
            # Skip builtin tools (code, subworkflow, mcp_server)
            if tool_id.startswith(BUILTIN_TOOL_PREFIXES):
                continue

            if tool_id not in tool_registry:
                result.add_error(
                    f"Tool '{tool_id}' not found for agent '{agent_id}'"
                )
            elif tool_registry[tool_id].get("status") == "deprecated":
                result.add_warning(
                    f"Tool '{tool_id}' is deprecated"
                )


def validate_io_contracts(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate agent I/O contracts (A2A safety)."""
    produced_keys = set()
    state_keys = set(workflow.get("state_schema", {}).keys())

    # Collect all produced state keys
    for agent_id in workflow.get("agents", []):
        if agent_id not in agent_registry:
            continue

        agent = agent_registry[agent_id]
        for key in agent.get("output_schema", []):
            if key in produced_keys:
                result.add_error(
                    f"State key '{key}' written by multiple agents"
                )
            produced_keys.add(key)

    # Check all required inputs are satisfied
    for agent_id in workflow.get("agents", []):
        if agent_id not in agent_registry:
            continue

        agent = agent_registry[agent_id]
        for key in agent.get("input_schema", []):
            if key not in produced_keys and key not in state_keys:
                result.add_error(
                    f"Agent '{agent_id}' expects '{key}' but no producer found"
                )


def validate_workflow_topology(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate workflow topology (no dead ends, no infinite loops)."""
    nodes = set(workflow.get("agents", []))
    connections = workflow.get("connections", [])

    # Build graph
    connected = set()
    for edge in connections:
        from_node = edge.get("from")
        to_node = edge.get("to")
        connected.add(from_node)
        connected.add(to_node)

        # Check nodes exist
        if from_node not in nodes:
            result.add_error(f"Connection references unknown agent '{from_node}'")
        if to_node not in nodes:
            result.add_error(f"Connection references unknown agent '{to_node}'")

    # Warn about isolated nodes
    for node in nodes:
        if node not in connected:
            result.add_warning(f"Agent '{node}' is isolated in workflow")


def validate_execution_model(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate execution model specific rules."""
    mode = workflow.get("execution_model")
    agents = workflow.get("agents", [])
    connections = workflow.get("connections", [])

    if mode == "sequential":
        # Sequential must be linear
        if len(connections) != len(agents) - 1:
            result.add_error("Sequential workflow must be linear")

    elif mode == "parallel":
        # Parallel workflows should have a merge point or final aggregator
        # This is a soft check (warning)
        if len(connections) == 0:
            result.add_warning("Parallel workflow has no connections")

    elif mode == "hierarchical":
        # Hierarchical must have hierarchy block
        hierarchy = workflow.get("hierarchy")
        if not hierarchy:
            result.add_error("Hierarchical workflow missing hierarchy block")


def validate_hierarchical_rules(
    workflow: Dict[str, Any],
    agent_registry: Dict[str, Dict[str, Any]],
    result: ValidationResult
) -> None:
    """Validate hierarchical workflow rules."""
    if workflow.get("execution_model") != "hierarchical":
        return

    hierarchy = workflow.get("hierarchy")
    if not hierarchy:
        return  # Already caught by validate_execution_model

    master = hierarchy.get("master_agent")
    agents = workflow.get("agents", [])

    if master not in agents:
        result.add_error("Master agent not found in agent list")

    # Sub-agents cannot call each other directly in hierarchical mode
    for agent_id in agents:
        if agent_id == master:
            continue

        if agent_id not in agent_registry:
            continue

        agent = agent_registry[agent_id]
        permissions = agent.get("permissions", {})
        if permissions.get("can_call_agents", False):
            result.add_error(
                f"Sub-agent '{agent_id}' cannot call agents in hierarchical mode"
            )


def validate_hitl_rules(workflow: Dict[str, Any], result: ValidationResult) -> None:
    """Validate HITL configuration."""
    hitl = workflow.get("human_in_loop", {})
    if not hitl.get("enabled"):
        return

    agents = workflow.get("agents", [])
    for point in hitl.get("review_points", []):
        if point not in agents:
            result.add_error(
                f"HITL review point '{point}' is invalid"
            )
