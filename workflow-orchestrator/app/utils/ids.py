"""
ID Generation Utilities
Generates unique IDs for workflows, agents, executions, and runs
"""
import uuid
from datetime import datetime
from typing import Optional


def generate_uuid() -> str:
    """
    Generate UUID v4

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def generate_workflow_id(prefix: str = "wf") -> str:
    """
    Generate workflow ID

    Args:
        prefix: Optional prefix (default: "wf")

    Returns:
        Workflow ID (e.g., "wf_abc123def456")

    Examples:
        >>> generate_workflow_id()
        "wf_7f3b4c2a1d8e"
    """
    uuid_short = uuid.uuid4().hex[:12]
    return f"{prefix}_{uuid_short}"


def generate_agent_id(prefix: str = "agent") -> str:
    """
    Generate agent ID

    Args:
        prefix: Optional prefix (default: "agent")

    Returns:
        Agent ID (e.g., "agent_abc123def456")
    """
    uuid_short = uuid.uuid4().hex[:12]
    return f"{prefix}_{uuid_short}"


def generate_execution_id(workflow_id: Optional[str] = None) -> str:
    """
    Generate execution ID

    Args:
        workflow_id: Optional workflow ID to include

    Returns:
        Execution ID

    Examples:
        >>> generate_execution_id("wf_123")
        "exec_wf_123_7f3b4c2a1d8e_20260109"
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uuid_short = uuid.uuid4().hex[:8]

    if workflow_id:
        return f"exec_{workflow_id}_{uuid_short}_{timestamp}"
    else:
        return f"exec_{uuid_short}_{timestamp}"


def generate_run_id() -> str:
    """
    Generate run ID (for individual agent/tool runs)

    Returns:
        Run ID

    Examples:
        >>> generate_run_id()
        "run_7f3b4c2a_20260109153045"
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uuid_short = uuid.uuid4().hex[:8]
    return f"run_{uuid_short}_{timestamp}"


def generate_checkpoint_id(execution_id: str, step: int) -> str:
    """
    Generate checkpoint ID

    Args:
        execution_id: Execution ID
        step: Step number

    Returns:
        Checkpoint ID

    Examples:
        >>> generate_checkpoint_id("exec_123", 5)
        "ckpt_exec_123_step_5"
    """
    return f"ckpt_{execution_id}_step_{step}"


def generate_tool_call_id() -> str:
    """
    Generate tool call ID

    Returns:
        Tool call ID
    """
    uuid_short = uuid.uuid4().hex[:8]
    return f"tool_{uuid_short}"


def extract_workflow_id_from_execution(execution_id: str) -> Optional[str]:
    """
    Extract workflow ID from execution ID

    Args:
        execution_id: Execution ID

    Returns:
        Workflow ID or None

    Examples:
        >>> extract_workflow_id_from_execution("exec_wf_123_abc_20260109")
        "wf_123"
    """
    parts = execution_id.split("_")
    if len(parts) >= 3 and parts[0] == "exec":
        # Format: exec_wf_123_abc_timestamp
        # Extract wf_123
        return f"{parts[1]}_{parts[2]}"
    return None


def is_valid_uuid(value: str) -> bool:
    """
    Check if string is valid UUID

    Args:
        value: String to check

    Returns:
        True if valid UUID
    """
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def generate_id() -> str:
    """
    Legacy: Generate generic ID

    Returns:
        UUID string
    """
    return generate_uuid()
