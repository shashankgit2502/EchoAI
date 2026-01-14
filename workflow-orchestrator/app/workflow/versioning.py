"""
Workflow Versioning
Version bumps, clone logic, and version management
"""
import re
from typing import Tuple, Optional
from pathlib import Path

from app.schemas.api_models import AgentSystemDesign
from app.core.logging import get_logger
from app.core.constants import WorkflowState

logger = get_logger(__name__)


# ============================================================================
# VERSION PARSING
# ============================================================================

def parse_version(version_string: str) -> Tuple[int, int]:
    """
    Parse version string into major and minor components

    Args:
        version_string: Version string (e.g., "1.0", "2.3")

    Returns:
        Tuple of (major, minor)

    Examples:
        >>> parse_version("1.0")
        (1, 0)
        >>> parse_version("3.5")
        (3, 5)
    """
    match = re.match(r'^(\d+)\.(\d+)$', version_string)

    if not match:
        logger.warning(f"Invalid version format: {version_string}, defaulting to 1.0")
        return (1, 0)

    major = int(match.group(1))
    minor = int(match.group(2))

    return (major, minor)


def format_version(major: int, minor: int) -> str:
    """
    Format version components into string

    Args:
        major: Major version
        minor: Minor version

    Returns:
        Version string

    Examples:
        >>> format_version(1, 0)
        "1.0"
        >>> format_version(2, 3)
        "2.3"
    """
    return f"{major}.{minor}"


# ============================================================================
# VERSION BUMPING
# ============================================================================

def bump_major_version(version_string: str) -> str:
    """
    Bump major version (X.Y -> X+1.0)

    Used when making breaking changes to workflow

    Args:
        version_string: Current version

    Returns:
        New version string

    Examples:
        >>> bump_major_version("1.0")
        "2.0"
        >>> bump_major_version("3.5")
        "4.0"
    """
    major, minor = parse_version(version_string)
    return format_version(major + 1, 0)


def bump_minor_version(version_string: str) -> str:
    """
    Bump minor version (X.Y -> X.Y+1)

    Used when making non-breaking changes to workflow

    Args:
        version_string: Current version

    Returns:
        New version string

    Examples:
        >>> bump_minor_version("1.0")
        "1.1"
        >>> bump_minor_version("2.3")
        "2.4"
    """
    major, minor = parse_version(version_string)
    return format_version(major, minor + 1)


def get_next_version(
    version_string: str,
    bump_type: str = "minor"
) -> str:
    """
    Get next version based on bump type

    Args:
        version_string: Current version
        bump_type: "major" or "minor" (default: "minor")

    Returns:
        New version string

    Raises:
        ValueError: If bump_type is invalid
    """
    if bump_type == "major":
        return bump_major_version(version_string)
    elif bump_type == "minor":
        return bump_minor_version(version_string)
    else:
        raise ValueError(f"Invalid bump_type: {bump_type}. Must be 'major' or 'minor'")


# Alias for backward compatibility
bump_version = get_next_version


# ============================================================================
# VERSION VALIDATION
# ============================================================================

def is_valid_version(version_string: str) -> bool:
    """
    Check if version string is valid

    Args:
        version_string: Version string to validate

    Returns:
        True if valid

    Examples:
        >>> is_valid_version("1.0")
        True
        >>> is_valid_version("3.5")
        True
        >>> is_valid_version("invalid")
        False
    """
    return bool(re.match(r'^\d+\.\d+$', version_string))


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings

    Args:
        version1: First version
        version2: Second version

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2

    Examples:
        >>> compare_versions("1.0", "2.0")
        -1
        >>> compare_versions("2.3", "2.3")
        0
        >>> compare_versions("3.5", "2.9")
        1
    """
    major1, minor1 = parse_version(version1)
    major2, minor2 = parse_version(version2)

    if major1 < major2:
        return -1
    elif major1 > major2:
        return 1
    else:
        # Major versions equal, compare minor
        if minor1 < minor2:
            return -1
        elif minor1 > minor2:
            return 1
        else:
            return 0


# ============================================================================
# WORKFLOW CLONING
# ============================================================================

def clone_workflow(
    workflow: AgentSystemDesign,
    new_name: Optional[str] = None,
    reset_version: bool = True,
    new_state: Optional[WorkflowState] = WorkflowState.DRAFT
) -> AgentSystemDesign:
    """
    Clone workflow with optional modifications

    Used for:
    - Creating new workflow from existing one
    - Editing FINAL workflows (must clone to edit)
    - Creating workflow variants

    Args:
        workflow: Source workflow
        new_name: New workflow name (defaults to "{original}_clone")
        reset_version: Reset version to 1.0 (default: True)
        new_state: New workflow state (default: DRAFT)

    Returns:
        Cloned workflow

    Examples:
        # Clone with new name
        cloned = clone_workflow(workflow, "my_new_workflow")

        # Clone for editing
        draft = clone_workflow(workflow, new_state=WorkflowState.DRAFT)
    """
    logger.info(f"Cloning workflow: {workflow.system_name}")

    # Determine new name
    if new_name is None:
        new_name = f"{workflow.system_name}_clone"

    # Create clone
    cloned_data = workflow.dict()
    cloned_data["system_name"] = new_name

    # Reset version if requested
    if reset_version:
        cloned_data["metadata"]["version"] = "1.0"

    # Update state
    if new_state:
        cloned_data["metadata"]["state"] = new_state.value

    # Create new workflow instance
    cloned_workflow = AgentSystemDesign(**cloned_data)

    logger.info(
        f"Cloned workflow: {workflow.system_name} -> {new_name} "
        f"(version: {cloned_workflow.metadata.version}, state: {new_state})"
    )

    return cloned_workflow


def create_editable_copy(workflow: AgentSystemDesign) -> AgentSystemDesign:
    """
    Create editable copy of FINAL workflow

    FINAL workflows are immutable - to edit them, create a DRAFT copy

    Args:
        workflow: FINAL workflow

    Returns:
        DRAFT copy of workflow

    Raises:
        ValueError: If workflow is not FINAL
    """
    if workflow.metadata.state != WorkflowState.FINAL.value:
        raise ValueError(
            f"Cannot create editable copy: workflow is not FINAL "
            f"(state: {workflow.metadata.state})"
        )

    # Clone as draft with version bump
    draft = clone_workflow(
        workflow,
        new_name=f"{workflow.system_name}_draft",
        reset_version=False,  # Keep version but will bump
        new_state=WorkflowState.DRAFT
    )

    # Bump minor version
    draft.metadata.version = bump_minor_version(draft.metadata.version)

    logger.info(
        f"Created editable copy: {workflow.system_name} "
        f"(v{workflow.metadata.version} -> v{draft.metadata.version})"
    )

    return draft


# ============================================================================
# VERSION HELPERS
# ============================================================================

def get_version(workflow: AgentSystemDesign) -> str:
    """
    Get workflow version

    Args:
        workflow: Workflow

    Returns:
        Version string
    """
    return workflow.metadata.version


def set_version(workflow: AgentSystemDesign, version: str) -> AgentSystemDesign:
    """
    Set workflow version

    Args:
        workflow: Workflow
        version: New version

    Returns:
        Updated workflow

    Raises:
        ValueError: If version is invalid
    """
    if not is_valid_version(version):
        raise ValueError(f"Invalid version format: {version}")

    workflow.metadata.version = version

    logger.debug(f"Set version: {workflow.system_name} -> {version}")

    return workflow


def increment_version(
    workflow: AgentSystemDesign,
    bump_type: str = "minor"
) -> AgentSystemDesign:
    """
    Increment workflow version

    Args:
        workflow: Workflow
        bump_type: "major" or "minor"

    Returns:
        Updated workflow
    """
    current_version = get_version(workflow)
    new_version = get_next_version(current_version, bump_type)

    return set_version(workflow, new_version)


# ============================================================================
# VERSION HISTORY
# ============================================================================

def extract_version_from_filename(filename: str) -> Optional[str]:
    """
    Extract version from workflow filename

    Args:
        filename: Workflow filename (e.g., "my_workflow_v2.3.json")

    Returns:
        Version string or None if not found

    Examples:
        >>> extract_version_from_filename("my_workflow_v2.3.json")
        "2.3"
        >>> extract_version_from_filename("my_workflow.json")
        None
    """
    match = re.search(r'_v(\d+\.\d+)', filename)

    if match:
        return match.group(1)

    return None


def build_versioned_filename(
    workflow_name: str,
    version: str,
    extension: str = ".json"
) -> str:
    """
    Build filename with version

    Args:
        workflow_name: Workflow name
        version: Version string
        extension: File extension (default: ".json")

    Returns:
        Versioned filename

    Examples:
        >>> build_versioned_filename("my_workflow", "2.3")
        "my_workflow_v2.3.json"
    """
    # Remove existing version suffix if present
    name_without_version = re.sub(r'_v\d+\.\d+$', '', workflow_name)

    return f"{name_without_version}_v{version}{extension}"
