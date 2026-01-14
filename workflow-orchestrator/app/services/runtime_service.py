"""
Runtime Service
Service layer wrapper for workflow execution
Provides clean interface for internal API communication
"""
from typing import Optional, Dict, Any, List, AsyncIterator
from app.schemas.api_models import (
    RuntimeExecuteRequest,
    RuntimeResumeRequest,
    ExecutionStatus,
    RuntimeMetrics,
    CheckpointInfo,
    ExecuteWorkflowRequest
)
from app.runtime.executor import WorkflowExecutor, ExecutionManager
from app.runtime.checkpoints import get_checkpoint_manager
from app.core.logging import get_logger

logger = get_logger(__name__)


class RuntimeService:
    """
    Service layer for runtime execution

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Async + idempotent
    - Ready for microservice extraction

    Responsibilities:
    - Workflow execution (test/final modes)
    - Execution resumption (HITL)
    - Checkpoint management
    - Metrics collection
    """

    def __init__(self):
        """Initialize runtime service"""
        self._executor = WorkflowExecutor()
        self._execution_manager = ExecutionManager()
        self._checkpoint_manager = get_checkpoint_manager()  # Uses singleton with default in-memory checkpointer
        logger.info("RuntimeService initialized")

    async def execute_workflow(
        self,
        request: RuntimeExecuteRequest
    ) -> ExecutionStatus:
        """
        Execute workflow (test or final mode)

        Args:
            request: Execution request

        Returns:
            ExecutionStatus with run details
        """
        logger.info(f"Executing workflow: {request.workflow_id} (mode: {request.execution_mode})")

        # Convert to internal format
        exec_request = ExecuteWorkflowRequest(
            workflow_id=request.workflow_id,
            execution_mode=request.execution_mode,
            version=request.version,
            input_payload=request.input_payload,
            thread_id=request.thread_id
        )

        result = await self._executor.execute(exec_request)

        logger.info(f"Execution completed: {result.run_id} (status: {result.status})")
        return result

    async def resume_execution(
        self,
        request: RuntimeResumeRequest
    ) -> ExecutionStatus:
        """
        Resume paused execution (HITL)

        Args:
            request: Resume request with human decision

        Returns:
            ExecutionStatus with updated status
        """
        logger.info(f"Resuming execution: {request.workflow_id} (thread: {request.thread_id})")

        result = await self._executor.resume_execution(
            request.workflow_id,
            request.thread_id,
            request.human_decision
        )

        logger.info(f"Execution resumed: {result.run_id}")
        return result

    def get_execution_metrics(
        self,
        run_id: str
    ) -> Optional[RuntimeMetrics]:
        """
        Get metrics for specific execution

        Args:
            run_id: Execution run ID

        Returns:
            RuntimeMetrics or None
        """
        logger.info(f"Getting metrics for run: {run_id}")

        metrics = self._execution_manager.get_metrics(run_id)
        return metrics

    def get_checkpoint(
        self,
        thread_id: str
    ) -> Optional[CheckpointInfo]:
        """
        Get checkpoint for thread

        Args:
            thread_id: Thread identifier

        Returns:
            CheckpointInfo or None
        """
        logger.info(f"Getting checkpoint: {thread_id}")

        checkpoint = self._checkpoint_manager.get_checkpoint(thread_id)
        if not checkpoint:
            return None

        return CheckpointInfo(
            thread_id=thread_id,
            checkpoint_id=checkpoint.get("checkpoint_id", ""),
            agent_id=checkpoint.get("agent_id", ""),
            state_snapshot=checkpoint.get("state", {}),
            created_at=checkpoint.get("created_at")
        )

    async def cancel_execution(
        self,
        run_id: str
    ) -> bool:
        """
        Cancel running execution

        Args:
            run_id: Execution run ID

        Returns:
            True if cancelled, False otherwise
        """
        logger.info(f"Cancelling execution: {run_id}")
        return await self._executor.cancel_execution(run_id)

    def get_execution_status(
        self,
        run_id: str
    ) -> Optional[ExecutionStatus]:
        """
        Get execution status by run ID

        Args:
            run_id: Execution run ID

        Returns:
            ExecutionStatus or None
        """
        logger.info(f"Getting execution status: {run_id}")
        return self._executor.get_execution_status(run_id)

    def list_active_executions(self) -> List[ExecutionStatus]:
        """
        List all active (running) executions

        Returns:
            List of active ExecutionStatus
        """
        logger.info("Listing active executions")
        return self._executor.list_active_executions()

    async def execute_streaming(
        self,
        request: RuntimeExecuteRequest
    ) -> AsyncIterator[ExecutionStatus]:
        """
        Execute workflow with streaming updates

        Args:
            request: Execution request

        Yields:
            ExecutionStatus updates as workflow progresses
        """
        logger.info(f"Starting streaming execution: {request.workflow_id}")

        # Convert to internal format
        exec_request = ExecuteWorkflowRequest(
            workflow_id=request.workflow_id,
            execution_mode=request.execution_mode,
            version=request.version,
            input_payload=request.input_payload,
            thread_id=request.thread_id
        )

        async for status in self._executor.execute_streaming(exec_request):
            yield status

    async def execute_batch(
        self,
        requests: List[RuntimeExecuteRequest]
    ) -> List[ExecutionStatus]:
        """
        Execute multiple workflows in parallel

        Args:
            requests: List of execution requests

        Returns:
            List of ExecutionStatus results
        """
        logger.info(f"Executing batch of {len(requests)} workflows")

        # Convert to internal format
        exec_requests = [
            ExecuteWorkflowRequest(
                workflow_id=req.workflow_id,
                execution_mode=req.execution_mode,
                version=req.version,
                input_payload=req.input_payload,
                thread_id=req.thread_id
            )
            for req in requests
        ]

        results = await self._execution_manager.execute_batch(exec_requests)
        logger.info(f"Batch execution completed: {len(results)} results")
        return results

    def get_execution_history(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionStatus]:
        """
        Get execution history

        Args:
            workflow_id: Optional workflow ID filter
            limit: Maximum number of results

        Returns:
            List of ExecutionStatus (most recent first)
        """
        logger.info(f"Getting execution history (workflow_id: {workflow_id}, limit: {limit})")
        return self._execution_manager.get_execution_history(workflow_id, limit)


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_runtime_service: Optional[RuntimeService] = None


def get_runtime_service() -> RuntimeService:
    """
    Get singleton runtime service instance

    Returns:
        RuntimeService instance
    """
    global _runtime_service

    if _runtime_service is None:
        _runtime_service = RuntimeService()

    return _runtime_service
