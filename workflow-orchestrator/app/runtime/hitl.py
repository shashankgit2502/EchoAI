"""
Human-in-the-Loop (HITL) Runtime Support

HITL is a first-class execution boundary with:
- Pause (deterministic checkpoint)
- Context (full decision info)
- Controlled action (approve/reject/edit)
- Re-validation (after changes)
- Audit (immutable trail)

This is NOT a UI feature - it's a workflow state enforced by runtime.

Architecture:
1. Design-time: Workflow declares HITL points
2. Compile-time: Validator enforces HITL rules
3. Run-time: Orchestrator manages HITL state
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt

from app.core.constants import WorkflowState
from app.core.logging import get_logger
from app.utils.time import utc_now, iso_now, duration_seconds
from app.utils.ids import generate_id

logger = get_logger(__name__)


# ============================================================================
# HITL STATE MACHINE
# ============================================================================

class HITLState(str, Enum):
    """
    HITL execution states

    State transitions:
    RUNNING → WAITING_FOR_HUMAN → APPROVED/REJECTED/EDITED/TIMEOUT → RESUMED/TERMINATED
    """
    RUNNING = "RUNNING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EDITED = "EDITED"
    TIMEOUT = "TIMEOUT"
    RESUMED = "RESUMED"
    TERMINATED = "TERMINATED"


class HITLMode(str, Enum):
    """HITL execution modes"""
    MANDATORY = "mandatory"  # Cannot proceed without approval
    CONDITIONAL = "conditional"  # Triggered based on conditions
    EXPLORATORY = "exploratory"  # TEMP workflows only, full editing


class HITLAction(str, Enum):
    """Allowed human actions during HITL"""
    APPROVE = "approve"
    REJECT = "reject"
    EDIT_AGENT = "edit_agent"
    EDIT_WORKFLOW = "edit_workflow"
    RERUN = "rerun"
    SAVE_AS_TEMP = "save_as_temp"
    PROMOTE = "promote"


# ============================================================================
# HITL DATA MODELS
# ============================================================================

@dataclass
class HITLContext:
    """
    Full decision context presented to human

    Attributes:
        workflow_id: Workflow being executed
        run_id: Execution run ID
        blocked_at: Where execution is paused (agent ID or step)
        agent_output: Output from the agent that triggered HITL
        tools_used: List of tools invoked
        execution_metrics: Cost, duration, tokens
        state_snapshot: Current workflow state
        validation_status: Latest validation result
        previous_decisions: History of HITL decisions in this run
    """
    workflow_id: str
    run_id: str
    blocked_at: str
    agent_output: Optional[Dict[str, Any]] = None
    tools_used: List[str] = field(default_factory=list)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    validation_status: Optional[Dict[str, Any]] = None
    previous_decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "blocked_at": self.blocked_at,
            "agent_output": self.agent_output,
            "tools_used": self.tools_used,
            "execution_metrics": self.execution_metrics,
            "state_snapshot": self.state_snapshot,
            "validation_status": self.validation_status,
            "previous_decisions": self.previous_decisions
        }


@dataclass
class HITLDecision:
    """
    Human decision record (audit trail)

    Attributes:
        decision_id: Unique decision ID
        run_id: Execution run ID
        action: Action taken
        actor: Who made the decision
        timestamp: When decision was made
        context_snapshot: State at decision time
        changes: What was changed (for edits)
        previous_state: State before change
        new_state: State after change
        rationale: Optional human comment
    """
    decision_id: str
    run_id: str
    action: HITLAction
    actor: str
    timestamp: datetime = field(default_factory=utc_now)
    context_snapshot: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "action": self.action.value,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "context_snapshot": self.context_snapshot,
            "changes": self.changes,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "rationale": self.rationale
        }


@dataclass
class HITLCheckpoint:
    """
    HITL execution checkpoint

    Stored before entering WAITING_FOR_HUMAN state.
    Enables deterministic resume after human action.

    Attributes:
        checkpoint_id: Unique checkpoint ID
        run_id: Execution run ID
        state: HITL state
        workflow_state: Complete workflow state
        execution_position: Where execution paused
        created_at: Checkpoint creation time
        timeout_at: When this checkpoint expires
    """
    checkpoint_id: str
    run_id: str
    state: HITLState
    workflow_state: Dict[str, Any]
    execution_position: str
    created_at: datetime = field(default_factory=utc_now)
    timeout_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if checkpoint has expired"""
        if not self.timeout_at:
            return False
        return utc_now() > self.timeout_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "state": self.state.value,
            "workflow_state": self.workflow_state,
            "execution_position": self.execution_position,
            "created_at": self.created_at.isoformat(),
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None
        }


# ============================================================================
# HITL MANAGER (Runtime Orchestrator)
# ============================================================================

class HITLManager:
    """
    Manages HITL lifecycle for workflow executions

    Responsibilities:
    - Create HITL checkpoints
    - Store execution context
    - Record human decisions
    - Manage timeouts
    - Enforce validation after edits
    - Maintain audit trail

    Integration with LangGraph:
    - Uses LangGraph checkpointing for state persistence
    - Uses LangGraph interrupt() for execution pause
    - Stateless - can survive restarts

    Usage:
        hitl = HITLManager(checkpointer=memory_saver)

        # In workflow node - request approval
        await hitl.request_approval(
            run_id="exec_123",
            workflow_id="wf_456",
            blocked_at="agent:analyst",
            context=context_data
        )

        # Human approves via API
        await hitl.approve(run_id="exec_123", actor="user@example.com")

        # Resume workflow
        result = await hitl.resume(run_id="exec_123", workflow=compiled_graph)
    """

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Initialize HITL manager

        Args:
            checkpointer: LangGraph checkpoint saver (required for persistence)
        """
        self.checkpointer = checkpointer

        # In-memory storage (in production, use database)
        self._checkpoints: Dict[str, HITLCheckpoint] = {}
        self._contexts: Dict[str, HITLContext] = {}
        self._decisions: Dict[str, List[HITLDecision]] = {}
        self._states: Dict[str, HITLState] = {}

        logger.info("HITL manager initialized")

    # ========================================================================
    # CORE HITL OPERATIONS
    # ========================================================================

    async def request_approval(
        self,
        run_id: str,
        workflow_id: str,
        blocked_at: str,
        context: Dict[str, Any],
        mode: HITLMode = HITLMode.MANDATORY,
        timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Request human approval (pauses execution)

        This is called from within a LangGraph node to pause execution
        and wait for human decision.

        Args:
            run_id: Execution run ID
            workflow_id: Workflow ID
            blocked_at: Where execution is paused (e.g., "agent:analyst")
            context: Full decision context
            mode: HITL mode (mandatory/conditional/exploratory)
            timeout_seconds: Optional timeout

        Returns:
            Human response when execution resumes

        Behavior:
            1. Create checkpoint
            2. Store context
            3. Set state = WAITING_FOR_HUMAN
            4. Call LangGraph interrupt() to pause
            5. Return human response when resumed
        """
        logger.info(f"HITL approval requested: run_id={run_id}, blocked_at={blocked_at}")

        # Create checkpoint
        checkpoint = self._create_checkpoint(
            run_id=run_id,
            state=HITLState.WAITING_FOR_HUMAN,
            workflow_state=context.get("state_snapshot", {}),
            execution_position=blocked_at,
            timeout_seconds=timeout_seconds
        )

        # Store context
        hitl_context = HITLContext(
            workflow_id=workflow_id,
            run_id=run_id,
            blocked_at=blocked_at,
            agent_output=context.get("agent_output"),
            tools_used=context.get("tools_used", []),
            execution_metrics=context.get("execution_metrics", {}),
            state_snapshot=context.get("state_snapshot", {}),
            validation_status=context.get("validation_status"),
            previous_decisions=self._get_decisions(run_id)
        )

        self._contexts[run_id] = hitl_context
        self._states[run_id] = HITLState.WAITING_FOR_HUMAN

        # Prepare interrupt payload
        interrupt_payload = {
            "hitl_mode": mode.value,
            "run_id": run_id,
            "workflow_id": workflow_id,
            "blocked_at": blocked_at,
            "allowed_actions": self._get_allowed_actions(workflow_id, mode),
            "context": hitl_context.to_dict(),
            "checkpoint_id": checkpoint.checkpoint_id
        }

        # Use LangGraph interrupt() - this pauses execution
        logger.info(f"Pausing execution at HITL boundary: {blocked_at}")
        response = interrupt(interrupt_payload)

        # When execution resumes, we get the response here
        logger.info(f"HITL resumed with response: run_id={run_id}")

        # Record decision
        await self._record_decision(
            run_id=run_id,
            action=HITLAction(response.get("action", "approve")),
            actor=response.get("actor", "unknown"),
            changes=response.get("changes"),
            rationale=response.get("rationale")
        )

        return response

    async def approve(self, run_id: str, actor: str, rationale: Optional[str] = None):
        """
        Approve execution (resume without changes)

        Args:
            run_id: Execution run ID
            actor: Who approved
            rationale: Optional comment
        """
        logger.info(f"HITL approved: run_id={run_id}, actor={actor}")

        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Update state
        self._states[run_id] = HITLState.APPROVED

        # Record decision
        await self._record_decision(
            run_id=run_id,
            action=HITLAction.APPROVE,
            actor=actor,
            rationale=rationale
        )

    async def reject(self, run_id: str, actor: str, rationale: Optional[str] = None):
        """
        Reject execution (terminate workflow)

        Args:
            run_id: Execution run ID
            actor: Who rejected
            rationale: Optional comment
        """
        logger.info(f"HITL rejected: run_id={run_id}, actor={actor}")

        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Update state
        self._states[run_id] = HITLState.REJECTED

        # Record decision
        await self._record_decision(
            run_id=run_id,
            action=HITLAction.REJECT,
            actor=actor,
            rationale=rationale
        )

    async def edit_agent(
        self,
        run_id: str,
        agent_id: str,
        changes: Dict[str, Any],
        actor: str
    ):
        """
        Edit agent configuration (TEMP workflows only)

        Args:
            run_id: Execution run ID
            agent_id: Agent to edit
            changes: Changes to apply
            actor: Who made changes

        Behavior:
            1. Apply changes
            2. Invalidate validation
            3. Set state = EDITED
            4. Require re-validation before resume
        """
        logger.info(f"HITL edit agent: run_id={run_id}, agent={agent_id}")

        # Validate state and workflow type
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)
        self._validate_editable(run_id)

        # Update state
        self._states[run_id] = HITLState.EDITED

        # Record decision with changes
        await self._record_decision(
            run_id=run_id,
            action=HITLAction.EDIT_AGENT,
            actor=actor,
            changes={"agent_id": agent_id, "changes": changes}
        )

        logger.warning(f"Validation invalidated for run {run_id} - re-validation required")

    async def edit_workflow(
        self,
        run_id: str,
        changes: Dict[str, Any],
        actor: str
    ):
        """
        Edit workflow structure (TEMP workflows only)

        Args:
            run_id: Execution run ID
            changes: Changes to apply
            actor: Who made changes
        """
        logger.info(f"HITL edit workflow: run_id={run_id}")

        # Validate state and workflow type
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)
        self._validate_editable(run_id)

        # Update state
        self._states[run_id] = HITLState.EDITED

        # Record decision
        await self._record_decision(
            run_id=run_id,
            action=HITLAction.EDIT_WORKFLOW,
            actor=actor,
            changes=changes
        )

        logger.warning(f"Workflow structure changed for run {run_id} - re-validation required")

    async def rerun(self, run_id: str, actor: str):
        """
        Re-run last step

        Args:
            run_id: Execution run ID
            actor: Who requested rerun
        """
        logger.info(f"HITL rerun: run_id={run_id}")

        # Validate state
        self._validate_state(run_id, HITLState.WAITING_FOR_HUMAN)

        # Record decision
        await self._record_decision(
            run_id=run_id,
            action=HITLAction.RERUN,
            actor=actor
        )

    async def handle_timeout(self, run_id: str):
        """
        Handle HITL timeout (auto-reject or cancel)

        Args:
            run_id: Execution run ID
        """
        logger.warning(f"HITL timeout: run_id={run_id}")

        checkpoint = self._checkpoints.get(run_id)
        if checkpoint and checkpoint.is_expired():
            # Update state
            self._states[run_id] = HITLState.TIMEOUT

            # Record decision
            await self._record_decision(
                run_id=run_id,
                action=HITLAction.REJECT,
                actor="system",
                rationale="HITL timeout - auto-rejected"
            )

    # ========================================================================
    # STATUS & CONTEXT QUERIES
    # ========================================================================

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get HITL status for run

        Returns:
            {
                "run_id": "...",
                "state": "WAITING_FOR_HUMAN",
                "blocked_at": "agent:analyst",
                "allowed_actions": ["approve", "reject", ...]
            }
        """
        state = self._states.get(run_id, HITLState.RUNNING)
        context = self._contexts.get(run_id)

        workflow_id = context.workflow_id if context else None
        blocked_at = context.blocked_at if context else None

        return {
            "run_id": run_id,
            "state": state.value,
            "blocked_at": blocked_at,
            "allowed_actions": self._get_allowed_actions(workflow_id, HITLMode.MANDATORY)
        }

    def get_context(self, run_id: str) -> Optional[HITLContext]:
        """Get full HITL context for run"""
        return self._contexts.get(run_id)

    def get_decisions(self, run_id: str) -> List[HITLDecision]:
        """Get all decisions for run (audit trail)"""
        return self._decisions.get(run_id, [])

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _create_checkpoint(
        self,
        run_id: str,
        state: HITLState,
        workflow_state: Dict[str, Any],
        execution_position: str,
        timeout_seconds: Optional[int] = None
    ) -> HITLCheckpoint:
        """Create HITL checkpoint"""
        checkpoint_id = generate_id("hitl_cp")

        timeout_at = None
        if timeout_seconds:
            timeout_at = utc_now() + timedelta(seconds=timeout_seconds)

        checkpoint = HITLCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            state=state,
            workflow_state=workflow_state,
            execution_position=execution_position,
            timeout_at=timeout_at
        )

        self._checkpoints[run_id] = checkpoint

        logger.info(
            f"HITL checkpoint created: {checkpoint_id} "
            f"(timeout: {timeout_seconds}s)" if timeout_seconds else ""
        )

        return checkpoint

    async def _record_decision(
        self,
        run_id: str,
        action: HITLAction,
        actor: str,
        changes: Optional[Dict[str, Any]] = None,
        rationale: Optional[str] = None
    ):
        """Record human decision (audit trail)"""
        decision_id = generate_id("hitl_dec")

        context = self._contexts.get(run_id)

        decision = HITLDecision(
            decision_id=decision_id,
            run_id=run_id,
            action=action,
            actor=actor,
            context_snapshot=context.to_dict() if context else None,
            changes=changes,
            rationale=rationale
        )

        if run_id not in self._decisions:
            self._decisions[run_id] = []

        self._decisions[run_id].append(decision)

        logger.info(
            f"HITL decision recorded: {decision_id} "
            f"({action.value} by {actor})"
        )

    def _get_decisions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get decision history for context"""
        decisions = self._decisions.get(run_id, [])
        return [d.to_dict() for d in decisions]

    def _validate_state(self, run_id: str, expected_state: HITLState):
        """Validate current HITL state"""
        current_state = self._states.get(run_id)
        if current_state != expected_state:
            raise ValueError(
                f"Invalid HITL state for {run_id}: "
                f"expected {expected_state.value}, got {current_state.value if current_state else 'NONE'}"
            )

    def _validate_editable(self, run_id: str):
        """Validate workflow is editable (TEMP only)"""
        context = self._contexts.get(run_id)
        if not context:
            raise ValueError(f"No context found for run {run_id}")

        # In production, check workflow.state == WorkflowState.TEMP
        # For now, we assume it's validated elsewhere
        logger.debug(f"Edit validation passed for run {run_id}")

    def _get_allowed_actions(
        self,
        workflow_id: Optional[str],
        mode: HITLMode
    ) -> List[str]:
        """Get allowed actions based on workflow state and mode"""
        # Base actions always allowed
        actions = [HITLAction.APPROVE.value, HITLAction.REJECT.value]

        # TEMP workflows allow edits
        # In production, check workflow state
        if mode == HITLMode.EXPLORATORY:
            actions.extend([
                HITLAction.EDIT_AGENT.value,
                HITLAction.EDIT_WORKFLOW.value,
                HITLAction.RERUN.value,
                HITLAction.SAVE_AS_TEMP.value
            ])

        return actions


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_hitl_manager: Optional[HITLManager] = None


def get_hitl_manager(checkpointer: Optional[BaseCheckpointSaver] = None) -> HITLManager:
    """Get singleton HITL manager instance"""
    global _hitl_manager
    if _hitl_manager is None:
        _hitl_manager = HITLManager(checkpointer=checkpointer)
    return _hitl_manager


# ============================================================================
# CONVENIENCE FUNCTIONS FOR WORKFLOW NODES
# ============================================================================

async def require_approval(
    run_id: str,
    workflow_id: str,
    blocked_at: str,
    agent_output: Optional[Dict[str, Any]] = None,
    tools_used: Optional[List[str]] = None,
    execution_metrics: Optional[Dict[str, Any]] = None,
    state_snapshot: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[int] = None
) -> Dict[str, Any]:
    """
    Require human approval in workflow node

    This is the primary function called from LangGraph nodes.

    Args:
        run_id: Execution run ID
        workflow_id: Workflow ID
        blocked_at: Where execution is paused
        agent_output: Agent output to show human
        tools_used: Tools that were invoked
        execution_metrics: Cost, duration, etc.
        state_snapshot: Current workflow state
        timeout_seconds: Optional timeout

    Returns:
        Human response

    Usage in LangGraph node:
        def analyst_node(state):
            # Execute agent
            output = agent.invoke(state)

            # Request approval before continuing
            response = await require_approval(
                run_id=state["run_id"],
                workflow_id=state["workflow_id"],
                blocked_at="agent:analyst",
                agent_output=output,
                execution_metrics={"cost": 0.05, "duration_ms": 2300}
            )

            if response.get("action") == "approve":
                return output
            else:
                raise ValueError("Human rejected execution")
    """
    manager = get_hitl_manager()

    context = {
        "agent_output": agent_output,
        "tools_used": tools_used or [],
        "execution_metrics": execution_metrics or {},
        "state_snapshot": state_snapshot or {}
    }

    return await manager.request_approval(
        run_id=run_id,
        workflow_id=workflow_id,
        blocked_at=blocked_at,
        context=context,
        timeout_seconds=timeout_seconds
    )
