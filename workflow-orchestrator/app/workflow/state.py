"""
Workflow State Helpers
Shared state management for workflow execution
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging import get_logger
from app.core.constants import WorkflowState

logger = get_logger(__name__)


# ============================================================================
# EXECUTION STATE
# ============================================================================

@dataclass
class ExecutionState:
    """
    Runtime execution state for workflow

    Tracks:
    - Current step
    - Agent outputs
    - Intermediate results
    - Error state
    - Execution metadata
    """
    workflow_id: str
    execution_id: str
    current_step: int = 0
    completed_steps: List[int] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, paused

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "agent_outputs": self.agent_outputs,
            "intermediate_results": self.intermediate_results,
            "errors": self.errors,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status
        }


# ============================================================================
# STATE HELPER
# ============================================================================

class StateHelper:
    """
    Helper for managing workflow execution state

    Provides utilities for:
    - Initializing execution state
    - Updating step progress
    - Recording agent outputs
    - Managing errors
    - State transitions

    Usage:
        helper = StateHelper()

        # Initialize state
        state = helper.create_execution_state(workflow_id, execution_id)

        # Update progress
        helper.advance_step(state)
        helper.record_agent_output(state, agent_id, output)

        # Handle errors
        helper.record_error(state, agent_id, error_message)

        # Check state
        if helper.is_completed(state):
            # Process results
    """

    def __init__(self):
        """Initialize state helper"""
        logger.info("State helper initialized")

    def create_execution_state(
        self,
        workflow_id: str,
        execution_id: str
    ) -> ExecutionState:
        """
        Create new execution state

        Args:
            workflow_id: Workflow ID
            execution_id: Execution ID

        Returns:
            ExecutionState
        """
        state = ExecutionState(
            workflow_id=workflow_id,
            execution_id=execution_id,
            started_at=datetime.utcnow(),
            status="pending"
        )

        logger.info(f"Created execution state: {execution_id} for workflow {workflow_id}")

        return state

    def start_execution(self, state: ExecutionState) -> ExecutionState:
        """
        Mark execution as started

        Args:
            state: Execution state

        Returns:
            Updated state
        """
        state.status = "running"
        state.started_at = datetime.utcnow()

        logger.info(f"Started execution: {state.execution_id}")

        return state

    def advance_step(self, state: ExecutionState) -> ExecutionState:
        """
        Advance to next step

        Args:
            state: Execution state

        Returns:
            Updated state
        """
        state.completed_steps.append(state.current_step)
        state.current_step += 1

        logger.debug(
            f"Advanced to step {state.current_step} "
            f"(execution: {state.execution_id})"
        )

        return state

    def record_agent_output(
        self,
        state: ExecutionState,
        agent_id: str,
        output: Any
    ) -> ExecutionState:
        """
        Record agent output

        Args:
            state: Execution state
            agent_id: Agent ID
            output: Agent output

        Returns:
            Updated state
        """
        state.agent_outputs[agent_id] = output

        logger.debug(f"Recorded output from agent: {agent_id}")

        return state

    def record_intermediate_result(
        self,
        state: ExecutionState,
        key: str,
        value: Any
    ) -> ExecutionState:
        """
        Record intermediate result

        Args:
            state: Execution state
            key: Result key
            value: Result value

        Returns:
            Updated state
        """
        state.intermediate_results[key] = value

        logger.debug(f"Recorded intermediate result: {key}")

        return state

    def record_error(
        self,
        state: ExecutionState,
        agent_id: str,
        error_message: str,
        error_type: Optional[str] = None
    ) -> ExecutionState:
        """
        Record error

        Args:
            state: Execution state
            agent_id: Agent that encountered error
            error_message: Error message
            error_type: Error type (optional)

        Returns:
            Updated state
        """
        error = {
            "agent_id": agent_id,
            "message": error_message,
            "type": error_type or "runtime_error",
            "timestamp": datetime.utcnow().isoformat(),
            "step": state.current_step
        }

        state.errors.append(error)

        logger.warning(
            f"Recorded error in execution {state.execution_id}: "
            f"{agent_id} - {error_message}"
        )

        return state

    def complete_execution(
        self,
        state: ExecutionState,
        success: bool = True
    ) -> ExecutionState:
        """
        Mark execution as completed

        Args:
            state: Execution state
            success: Whether execution was successful

        Returns:
            Updated state
        """
        state.completed_at = datetime.utcnow()
        state.status = "completed" if success else "failed"

        logger.info(
            f"Completed execution: {state.execution_id} "
            f"(status: {state.status})"
        )

        return state

    def pause_execution(self, state: ExecutionState) -> ExecutionState:
        """
        Pause execution (for HITL)

        Args:
            state: Execution state

        Returns:
            Updated state
        """
        state.status = "paused"

        logger.info(f"Paused execution: {state.execution_id}")

        return state

    def resume_execution(self, state: ExecutionState) -> ExecutionState:
        """
        Resume paused execution

        Args:
            state: Execution state

        Returns:
            Updated state
        """
        state.status = "running"

        logger.info(f"Resumed execution: {state.execution_id}")

        return state

    def is_completed(self, state: ExecutionState) -> bool:
        """
        Check if execution is completed

        Args:
            state: Execution state

        Returns:
            True if completed
        """
        return state.status in ["completed", "failed"]

    def is_running(self, state: ExecutionState) -> bool:
        """
        Check if execution is running

        Args:
            state: Execution state

        Returns:
            True if running
        """
        return state.status == "running"

    def is_paused(self, state: ExecutionState) -> bool:
        """
        Check if execution is paused

        Args:
            state: Execution state

        Returns:
            True if paused
        """
        return state.status == "paused"

    def has_errors(self, state: ExecutionState) -> bool:
        """
        Check if execution has errors

        Args:
            state: Execution state

        Returns:
            True if has errors
        """
        return len(state.errors) > 0

    def get_agent_output(
        self,
        state: ExecutionState,
        agent_id: str
    ) -> Optional[Any]:
        """
        Get agent output

        Args:
            state: Execution state
            agent_id: Agent ID

        Returns:
            Agent output or None
        """
        return state.agent_outputs.get(agent_id)

    def get_intermediate_result(
        self,
        state: ExecutionState,
        key: str
    ) -> Optional[Any]:
        """
        Get intermediate result

        Args:
            state: Execution state
            key: Result key

        Returns:
            Result value or None
        """
        return state.intermediate_results.get(key)

    def get_execution_duration(self, state: ExecutionState) -> Optional[float]:
        """
        Get execution duration in seconds

        Args:
            state: Execution state

        Returns:
            Duration in seconds or None if not completed
        """
        if state.started_at and state.completed_at:
            duration = (state.completed_at - state.started_at).total_seconds()
            return duration

        return None

    def get_progress(
        self,
        state: ExecutionState,
        total_steps: int
    ) -> float:
        """
        Get execution progress percentage

        Args:
            state: Execution state
            total_steps: Total number of steps

        Returns:
            Progress percentage (0.0 to 1.0)
        """
        if total_steps == 0:
            return 0.0

        return len(state.completed_steps) / total_steps


# ============================================================================
# WORKFLOW STATE TRANSITIONS
# ============================================================================

def can_transition_state(
    current_state: WorkflowState,
    new_state: WorkflowState
) -> bool:
    """
    Check if state transition is valid

    Valid transitions:
    - DRAFT -> TEMP, FINAL
    - TEMP -> DRAFT, FINAL
    - FINAL -> ARCHIVED
    - Any state -> ARCHIVED

    Args:
        current_state: Current workflow state
        new_state: Desired new state

    Returns:
        True if transition is valid
    """
    # Allow transition to ARCHIVED from any state
    if new_state == WorkflowState.ARCHIVED:
        return True

    # Valid transitions
    valid_transitions = {
        WorkflowState.DRAFT: [WorkflowState.TEMP, WorkflowState.FINAL],
        WorkflowState.TEMP: [WorkflowState.DRAFT, WorkflowState.FINAL],
        WorkflowState.FINAL: [WorkflowState.ARCHIVED],
        WorkflowState.ARCHIVED: []  # Cannot transition from archived
    }

    return new_state in valid_transitions.get(current_state, [])


def get_valid_transitions(current_state: WorkflowState) -> List[WorkflowState]:
    """
    Get list of valid state transitions

    Args:
        current_state: Current workflow state

    Returns:
        List of valid target states
    """
    valid_transitions = {
        WorkflowState.DRAFT: [WorkflowState.TEMP, WorkflowState.FINAL, WorkflowState.ARCHIVED],
        WorkflowState.TEMP: [WorkflowState.DRAFT, WorkflowState.FINAL, WorkflowState.ARCHIVED],
        WorkflowState.FINAL: [WorkflowState.ARCHIVED],
        WorkflowState.ARCHIVED: []
    }

    return valid_transitions.get(current_state, [])


# ============================================================================
# GLOBAL HELPER INSTANCE
# ============================================================================

_global_helper: Optional[StateHelper] = None


def get_state_helper() -> StateHelper:
    """
    Get global state helper instance

    Singleton pattern

    Returns:
        StateHelper
    """
    global _global_helper

    if _global_helper is None:
        _global_helper = StateHelper()

    return _global_helper
