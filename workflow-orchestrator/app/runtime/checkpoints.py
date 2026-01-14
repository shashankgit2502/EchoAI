"""
Checkpoint Management
Manages workflow execution checkpoints for resume/recovery
Uses LangGraph's checkpoint system for state persistence
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.memory import MemorySaver

# SqliteSaver requires separate package: pip install langgraph-checkpoint-sqlite
try:
    from langgraph_checkpoint_sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    SqliteSaver = None  # Fallback to None if not installed

from app.core.logging import get_logger
from app.utils.time import utc_now
from app.utils.ids import generate_checkpoint_id

logger = get_logger(__name__)


# ============================================================================
# CHECKPOINT METADATA
# ============================================================================

@dataclass
class CheckpointMetadata:
    """
    Extended metadata for workflow checkpoints

    Attributes:
        checkpoint_id: Unique checkpoint ID
        run_id: Execution run ID
        workflow_id: Workflow ID
        step_number: Which step was checkpointed
        agent_id: Agent that was executing
        created_at: When checkpoint was created
        checkpoint_type: Type of checkpoint (auto, manual, hitl, error)
        state_size_bytes: Size of serialized state
        tags: Optional tags for filtering
    """
    checkpoint_id: str
    run_id: str
    workflow_id: str
    step_number: int
    agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    checkpoint_type: str = "auto"
    state_size_bytes: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "step_number": self.step_number,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "checkpoint_type": self.checkpoint_type,
            "state_size_bytes": self.state_size_bytes,
            "tags": self.tags
        }


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Manages workflow execution checkpoints

    Provides high-level checkpoint operations on top of LangGraph's
    checkpoint system. Handles:
    - Checkpoint creation with metadata
    - Checkpoint retrieval
    - Resume from checkpoint
    - Checkpoint cleanup
    - Checkpoint search/filtering

    Integration with LangGraph:
    - Uses LangGraph's BaseCheckpointSaver interface
    - Supports MemorySaver (development)
    - Supports SqliteSaver (production)
    - Can be extended to PostgreSQL, Redis, etc.

    Usage:
        # Initialize with checkpointer
        from langgraph_checkpoint_sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        manager = CheckpointManager(checkpointer)

        # Checkpoints are created automatically by LangGraph
        # Just configure the workflow with checkpointer:
        workflow = graph.compile(checkpointer=checkpointer)

        # Resume from checkpoint
        config = manager.get_resume_config(run_id="exec_123")
        result = workflow.invoke(input_data, config=config)

        # List checkpoints
        checkpoints = manager.list_checkpoints(workflow_id="wf_456")
    """

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        auto_cleanup: bool = True,
        max_checkpoints_per_run: int = 100
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpointer: LangGraph checkpoint saver
            auto_cleanup: Automatically cleanup old checkpoints
            max_checkpoints_per_run: Max checkpoints to keep per run
        """
        self.checkpointer = checkpointer
        self.auto_cleanup = auto_cleanup
        self.max_checkpoints_per_run = max_checkpoints_per_run

        # Metadata storage (in production, use database)
        self._metadata: Dict[str, CheckpointMetadata] = {}

        logger.info(
            f"Checkpoint manager initialized "
            f"(auto_cleanup={auto_cleanup}, max_per_run={max_checkpoints_per_run})"
        )

    # ========================================================================
    # CHECKPOINT CONFIGURATION
    # ========================================================================

    def get_checkpoint_config(
        self,
        run_id: str,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get checkpoint configuration for workflow execution

        Args:
            run_id: Execution run ID
            thread_id: Optional thread ID (defaults to run_id)

        Returns:
            LangGraph config dict

        Usage:
            config = manager.get_checkpoint_config(run_id="exec_123")
            result = workflow.invoke(input_data, config=config)
        """
        if not thread_id:
            thread_id = run_id

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": None
            }
        }

    def get_resume_config(
        self,
        run_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get config to resume from specific checkpoint

        Args:
            run_id: Execution run ID
            checkpoint_id: Optional specific checkpoint (default: latest)

        Returns:
            LangGraph config dict

        Usage:
            # Resume from latest checkpoint
            config = manager.get_resume_config(run_id="exec_123")
            result = workflow.invoke(resume_data, config=config)

            # Resume from specific checkpoint
            config = manager.get_resume_config(
                run_id="exec_123",
                checkpoint_id="cp_abc123"
            )
        """
        return {
            "configurable": {
                "thread_id": run_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id
            }
        }

    # ========================================================================
    # CHECKPOINT OPERATIONS
    # ========================================================================

    def record_checkpoint_metadata(
        self,
        checkpoint_id: str,
        run_id: str,
        workflow_id: str,
        step_number: int,
        agent_id: Optional[str] = None,
        checkpoint_type: str = "auto",
        tags: Optional[List[str]] = None
    ):
        """
        Record metadata for a checkpoint

        LangGraph creates checkpoints automatically.
        This method stores additional metadata about them.

        Args:
            checkpoint_id: Checkpoint ID
            run_id: Execution run ID
            workflow_id: Workflow ID
            step_number: Current step number
            agent_id: Agent being executed
            checkpoint_type: Type (auto, manual, hitl, error)
            tags: Optional tags
        """
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            workflow_id=workflow_id,
            step_number=step_number,
            agent_id=agent_id,
            checkpoint_type=checkpoint_type,
            tags=tags or []
        )

        self._metadata[checkpoint_id] = metadata

        logger.debug(
            f"Checkpoint metadata recorded: {checkpoint_id} "
            f"(run={run_id}, step={step_number}, type={checkpoint_type})"
        )

    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for checkpoint"""
        return self._metadata.get(checkpoint_id)

    def list_checkpoints(
        self,
        run_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        checkpoint_type: Optional[str] = None,
        limit: int = 50
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints with filtering

        Args:
            run_id: Filter by run ID
            workflow_id: Filter by workflow ID
            checkpoint_type: Filter by type
            limit: Max results

        Returns:
            List of checkpoint metadata
        """
        results = []

        for metadata in self._metadata.values():
            # Apply filters
            if run_id and metadata.run_id != run_id:
                continue
            if workflow_id and metadata.workflow_id != workflow_id:
                continue
            if checkpoint_type and metadata.checkpoint_type != checkpoint_type:
                continue

            results.append(metadata)

            if len(results) >= limit:
                break

        # Sort by created_at descending
        results.sort(key=lambda m: m.created_at, reverse=True)

        return results

    def get_latest_checkpoint(self, run_id: str) -> Optional[CheckpointMetadata]:
        """Get latest checkpoint for run"""
        checkpoints = self.list_checkpoints(run_id=run_id, limit=1)
        return checkpoints[0] if checkpoints else None

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def cleanup_old_checkpoints(
        self,
        run_id: str,
        keep_count: Optional[int] = None
    ):
        """
        Cleanup old checkpoints for a run

        Args:
            run_id: Execution run ID
            keep_count: How many to keep (default: max_checkpoints_per_run)
        """
        if keep_count is None:
            keep_count = self.max_checkpoints_per_run

        checkpoints = self.list_checkpoints(run_id=run_id, limit=1000)

        if len(checkpoints) <= keep_count:
            logger.debug(f"No cleanup needed for run {run_id}")
            return

        # Keep most recent, remove oldest
        to_remove = checkpoints[keep_count:]

        for metadata in to_remove:
            # Remove metadata
            if metadata.checkpoint_id in self._metadata:
                del self._metadata[metadata.checkpoint_id]

        logger.info(
            f"Cleaned up {len(to_remove)} old checkpoints for run {run_id} "
            f"(kept {keep_count})"
        )

    def cleanup_run(self, run_id: str):
        """Remove all checkpoints for a run"""
        checkpoints = self.list_checkpoints(run_id=run_id, limit=1000)

        for metadata in checkpoints:
            if metadata.checkpoint_id in self._metadata:
                del self._metadata[metadata.checkpoint_id]

        logger.info(f"Cleaned up all checkpoints for run {run_id}")


# ============================================================================
# CHECKPOINT SAVER FACTORY
# ============================================================================

def create_checkpointer(
    storage_type: str = "memory",
    storage_path: Optional[str] = None
) -> BaseCheckpointSaver:
    """
    Create checkpoint saver

    Args:
        storage_type: Type of storage ("memory", "sqlite", "postgres")
        storage_path: Path to storage file (for sqlite)

    Returns:
        BaseCheckpointSaver instance

    Usage:
        # Development - in-memory
        checkpointer = create_checkpointer("memory")

        # Production - SQLite
        checkpointer = create_checkpointer("sqlite", "checkpoints.db")

        # Use with workflow
        workflow = graph.compile(checkpointer=checkpointer)
    """
    if storage_type == "memory":
        logger.info("Creating in-memory checkpointer (dev only)")
        return MemorySaver()

    elif storage_type == "sqlite":
        if not SQLITE_AVAILABLE:
            raise ImportError(
                "SqliteSaver not available. Install with: pip install langgraph-checkpoint-sqlite"
            )

        if not storage_path:
            storage_path = "checkpoints.db"

        logger.info(f"Creating SQLite checkpointer: {storage_path}")

        # Ensure directory exists
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

        return SqliteSaver.from_conn_string(storage_path)

    elif storage_type == "postgres":
        # TODO: Implement PostgreSQL checkpointer
        # from langgraph.checkpoint.postgres import PostgresSaver
        # return PostgresSaver.from_conn_string(storage_path)
        raise NotImplementedError("PostgreSQL checkpointer not yet implemented")

    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> CheckpointManager:
    """
    Get singleton checkpoint manager

    Args:
        checkpointer: Optional checkpointer (uses memory if not provided)

    Returns:
        CheckpointManager instance
    """
    global _checkpoint_manager

    if _checkpoint_manager is None:
        if checkpointer is None:
            checkpointer = create_checkpointer("memory")

        _checkpoint_manager = CheckpointManager(checkpointer)

    return _checkpoint_manager


# ============================================================================
# EXAMPLE USAGE WITH LANGGRAPH
# ============================================================================

"""
Example: Using checkpoints with LangGraph workflow

from langgraph.graph import StateGraph
from langgraph_checkpoint_sqlite import SqliteSaver

# Create checkpointer
checkpointer = create_checkpointer("sqlite", "workflow_checkpoints.db")
manager = CheckpointManager(checkpointer)

# Define workflow
class WorkflowState(TypedDict):
    run_id: str
    workflow_id: str
    step: int
    data: Dict[str, Any]

def agent_node(state: WorkflowState):
    # Execute agent
    step = state["step"]
    logger.info(f"Executing step {step}")

    # Record checkpoint metadata (LangGraph creates checkpoint automatically)
    checkpoint_id = generate_checkpoint_id()
    manager.record_checkpoint_metadata(
        checkpoint_id=checkpoint_id,
        run_id=state["run_id"],
        workflow_id=state["workflow_id"],
        step_number=step,
        agent_id="agent_1",
        checkpoint_type="auto"
    )

    # Update state
    return {
        **state,
        "step": step + 1,
        "data": {"result": f"Step {step} complete"}
    }

# Build workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

# Compile with checkpointing
compiled = workflow.compile(checkpointer=checkpointer)

# Run workflow
config = manager.get_checkpoint_config(run_id="exec_123")
result = compiled.invoke(
    {
        "run_id": "exec_123",
        "workflow_id": "wf_456",
        "step": 0,
        "data": {}
    },
    config=config
)

# If workflow crashes, resume from checkpoint
resume_config = manager.get_resume_config(run_id="exec_123")
result = compiled.invoke(
    {"run_id": "exec_123"},  # Minimal input to resume
    config=resume_config
)
"""
