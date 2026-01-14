"""
Runtime Components
Phase 9: Guards, HITL, Checkpoints, Telemetry
"""
from app.runtime.guards import (
    GuardViolation,
    GuardResult,
    ExecutionContext,
    RuntimeGuard,
    check_guards,
    ExecutionGuardError
)

from app.runtime.hitl import (
    HITLState,
    HITLMode,
    HITLAction,
    HITLContext,
    HITLDecision,
    HITLCheckpoint,
    HITLManager,
    get_hitl_manager,
    require_approval
)

from app.runtime.checkpoints import (
    CheckpointMetadata,
    CheckpointManager,
    create_checkpointer,
    get_checkpoint_manager
)

from app.runtime.telemetry import (
    ExecutionSpan,
    ExecutionMetrics,
    RuntimeTelemetry,
    get_runtime_telemetry,
    track_execution,
    record_agent_execution,
    record_tool_execution
)

__all__ = [
    # Guards
    "GuardViolation",
    "GuardResult",
    "ExecutionContext",
    "RuntimeGuard",
    "check_guards",
    "ExecutionGuardError",

    # HITL
    "HITLState",
    "HITLMode",
    "HITLAction",
    "HITLContext",
    "HITLDecision",
    "HITLCheckpoint",
    "HITLManager",
    "get_hitl_manager",
    "require_approval",

    # Checkpoints
    "CheckpointMetadata",
    "CheckpointManager",
    "create_checkpointer",
    "get_checkpoint_manager",

    # Telemetry
    "ExecutionSpan",
    "ExecutionMetrics",
    "RuntimeTelemetry",
    "get_runtime_telemetry",
    "track_execution",
    "record_agent_execution",
    "record_tool_execution",
]
