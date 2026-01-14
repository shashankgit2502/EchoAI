"""
Workflow Runtime Executor
Executes compiled LangGraph workflows with HITL support
"""
import uuid
from typing import Dict, Any, Optional, List, AsyncIterator
from datetime import datetime
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError, GraphInterrupt

from app.schemas.api_models import (
    AgentSystemDesign,
    ExecuteWorkflowRequest,
    ExecutionStatus
)
from app.workflow.compiler import WorkflowCompiler, WorkflowState
from app.storage.filesystem import WorkflowStorage
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowExecutor:
    """
    Executes compiled LangGraph workflows

    Features:
    - Synchronous and asynchronous execution
    - HITL support via checkpointing
    - Streaming execution updates
    - Error handling and recovery
    - Execution state management
    """

    def __init__(self):
        self.settings = get_settings()
        self.compiler = WorkflowCompiler()
        self.storage = WorkflowStorage()

        # Track active executions
        self._active_executions: Dict[str, ExecutionStatus] = {}

    async def execute(
        self,
        request: ExecuteWorkflowRequest
    ) -> ExecutionStatus:
        """
        Execute a workflow (synchronous, returns final result)

        Args:
            request: Execution request with workflow_id, mode, version, input

        Returns:
            ExecutionStatus with final result
        """
        logger.info(f"Executing workflow: {request.workflow_id} (mode: {request.execution_mode})")

        # Generate run ID
        run_id = str(uuid.uuid4())

        # Generate thread_id for conversation continuation if not provided
        thread_id = request.thread_id or str(uuid.uuid4())

        # Load agent system
        agent_system = self._load_agent_system(request)

        # Compile workflow with checkpointing enabled for conversation continuation
        compiled_workflow = self._compile_workflow(
            agent_system,
            enable_checkpointing=True  # Always enable for conversation continuation
        )

        # Create initial state
        initial_state = self._create_initial_state(request.input_payload)

        # Initialize execution status
        status = ExecutionStatus(
            run_id=run_id,
            workflow_id=request.workflow_id,
            status="running",
            started_at=datetime.utcnow(),
            thread_id=thread_id
        )
        self._active_executions[run_id] = status

        try:
            # Always execute with checkpointing for conversation continuation
            final_state = await self._execute_with_checkpointing(
                compiled_workflow,
                initial_state,
                thread_id,
                run_id
            )

            # Extract output
            output = self._extract_output(final_state)

            # Update status
            if output.get("error"):
                status.status = "failed"
                status.error = output.get("error")
            else:
                status.status = "completed"
            status.output = output
            status.completed_at = datetime.utcnow()

            if status.status == "failed":
                logger.warning(f"Workflow {request.workflow_id} completed with errors (run_id: {run_id})")
            else:
                logger.info(f"Workflow {request.workflow_id} completed successfully (run_id: {run_id})")

            return status

        except GraphInterrupt as gi:
            # HITL interrupt - workflow paused for human input
            logger.info(f"Workflow {request.workflow_id} interrupted for HITL (run_id: {run_id})")

            status.status = "paused"
            status.current_step = str(gi)
            status.completed_at = datetime.utcnow()

            return status

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)

            status.status = "failed"
            status.error = str(e)
            status.completed_at = datetime.utcnow()

            return status

        finally:
            # Cleanup
            if run_id in self._active_executions:
                self._active_executions[run_id] = status

    async def execute_streaming(
        self,
        request: ExecuteWorkflowRequest
    ) -> AsyncIterator[ExecutionStatus]:
        """
        Execute workflow with streaming updates

        Yields ExecutionStatus updates as workflow progresses

        Args:
            request: Execution request

        Yields:
            ExecutionStatus updates
        """
        logger.info(f"Starting streaming execution: {request.workflow_id}")

        run_id = str(uuid.uuid4())
        thread_id = request.thread_id or str(uuid.uuid4())

        # Load and compile
        agent_system = self._load_agent_system(request)
        compiled_workflow = self._compile_workflow(
            agent_system,
            enable_checkpointing=True  # Always enable for conversation continuation
        )

        initial_state = self._create_initial_state(request.input_payload)

        # Initialize status
        status = ExecutionStatus(
            run_id=run_id,
            workflow_id=request.workflow_id,
            status="running",
            started_at=datetime.utcnow(),
            thread_id=thread_id
        )

        # Yield initial status
        yield status

        try:
            # Stream execution with thread_id for conversation continuation
            config = {"configurable": {"thread_id": thread_id}}

            async for state_update in compiled_workflow.astream(initial_state, config):
                # Extract current step
                current_agent = state_update.get("current_agent")
                if current_agent:
                    status.current_step = current_agent

                # Yield progress update
                yield status

            # Final state
            status.status = "completed"
            status.output = self._extract_output(state_update)
            status.completed_at = datetime.utcnow()

            yield status

            logger.info(f"Streaming execution completed: {run_id}")

        except GraphInterrupt as gi:
            status.status = "paused"
            status.current_step = str(gi)
            status.completed_at = datetime.utcnow()
            yield status

        except Exception as e:
            logger.error(f"Streaming execution failed: {e}", exc_info=True)
            status.status = "failed"
            status.error = str(e)
            status.completed_at = datetime.utcnow()
            yield status

    async def resume_execution(
        self,
        workflow_id: str,
        thread_id: str,
        human_input: Dict[str, Any]
    ) -> ExecutionStatus:
        """
        Resume a paused workflow with human input (HITL)

        Args:
            workflow_id: Workflow identifier
            thread_id: Thread ID from original execution
            human_input: Human feedback to resume with

        Returns:
            ExecutionStatus after resumption
        """
        logger.info(f"Resuming workflow {workflow_id} with thread {thread_id}")

        run_id = str(uuid.uuid4())

        # Load agent system
        agent_system = self.storage.load_temp(workflow_id)

        # Compile with checkpointing
        compiled_workflow = self._compile_workflow(agent_system, enable_checkpointing=True)

        # Resume from checkpoint with human input
        config = {
            "configurable": {"thread_id": thread_id}
        }

        status = ExecutionStatus(
            run_id=run_id,
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.utcnow(),
            thread_id=thread_id
        )

        try:
            # Resume execution
            # LangGraph will automatically load from checkpoint
            final_state = await self._execute_with_resume(
                compiled_workflow,
                human_input,
                config,
                run_id
            )

            output = self._extract_output(final_state)

            status.status = "completed"
            status.output = output
            status.completed_at = datetime.utcnow()

            logger.info(f"Resumed workflow completed: {run_id}")

            return status

        except GraphInterrupt as gi:
            # Another interrupt
            status.status = "paused"
            status.current_step = str(gi)
            status.completed_at = datetime.utcnow()
            return status

        except Exception as e:
            logger.error(f"Resume execution failed: {e}", exc_info=True)
            status.status = "failed"
            status.error = str(e)
            status.completed_at = datetime.utcnow()
            return status

    def get_execution_status(self, run_id: str) -> Optional[ExecutionStatus]:
        """Get status of an active or completed execution"""
        return self._active_executions.get(run_id)

    def list_active_executions(self) -> List[ExecutionStatus]:
        """List all active executions"""
        return [
            status for status in self._active_executions.values()
            if status.status == "running"
        ]

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _load_agent_system(self, request: ExecuteWorkflowRequest) -> AgentSystemDesign:
        """Load agent system based on execution mode"""
        if request.execution_mode == "test":
            # Load from TEMP
            return self.storage.load_temp(request.workflow_id)
        elif request.execution_mode == "final":
            # Load from FINAL
            return self.storage.load_final(request.workflow_id, request.version)
        else:
            raise ValueError(f"Invalid execution mode: {request.execution_mode}")

    def _compile_workflow(
        self,
        agent_system: AgentSystemDesign,
        enable_checkpointing: bool
    ) -> CompiledStateGraph:
        """Compile agent system to executable workflow"""
        logger.debug(f"Compiling workflow: {agent_system.system_name}")

        return self.compiler.compile(
            agent_system,
            workflow_name=None,  # Use first workflow
            enable_checkpointing=enable_checkpointing
        )

    def _create_initial_state(self, input_payload: Dict[str, Any]) -> WorkflowState:
        """Create initial workflow state as a proper dict for TypedDict compatibility"""
        return {
            "workflow_input": input_payload or {},
            "agent_outputs": {},
            "current_agent": None,
            "error": None,
            "metadata": {},
            "messages": [],
            "workflow_output": None
        }

    async def _execute_simple(
        self,
        compiled_workflow: CompiledStateGraph,
        initial_state: WorkflowState,
        run_id: str
    ) -> WorkflowState:
        """Execute workflow without checkpointing"""
        logger.debug(f"Executing workflow (simple mode): {run_id}")
        logger.debug(f"Initial state keys: {list(initial_state.keys()) if initial_state else 'None'}")

        # Execute workflow
        final_state = await compiled_workflow.ainvoke(initial_state)

        # Log the result for debugging
        if final_state is None:
            logger.warning(f"Workflow {run_id} returned None state")
        else:
            logger.debug(f"Final state type: {type(final_state)}")
            if isinstance(final_state, dict):
                logger.debug(f"Final state keys: {list(final_state.keys())}")
                agent_outputs = final_state.get("agent_outputs", {})
                logger.debug(f"Agent outputs count: {len(agent_outputs) if agent_outputs else 0}")

        return final_state

    async def _execute_with_checkpointing(
        self,
        compiled_workflow: CompiledStateGraph,
        initial_state: WorkflowState,
        thread_id: str,
        run_id: str
    ) -> WorkflowState:
        """Execute workflow with checkpointing for HITL"""
        logger.debug(f"Executing workflow (checkpointing mode): {run_id}, thread: {thread_id}")

        config = {
            "configurable": {"thread_id": thread_id}
        }

        # Execute with checkpointing
        final_state = await compiled_workflow.ainvoke(initial_state, config)

        return final_state

    async def _execute_with_resume(
        self,
        compiled_workflow: CompiledStateGraph,
        human_input: Dict[str, Any],
        config: Dict[str, Any],
        run_id: str
    ) -> WorkflowState:
        """Resume execution with human input"""
        logger.debug(f"Resuming execution: {run_id}")

        # Create state with human input (proper dict format for TypedDict)
        resume_state = {
            "workflow_input": human_input or {},
            "agent_outputs": {},
            "current_agent": None,
            "error": None,
            "metadata": {"resumed": True},
            "messages": [],
            "workflow_output": None
        }

        # Resume execution (LangGraph loads from checkpoint)
        final_state = await compiled_workflow.ainvoke(resume_state, config)

        return final_state

    def _extract_output(self, final_state: WorkflowState) -> Dict[str, Any]:
        """
        Extract final output from workflow state

        Priority:
        1. workflow_output if set
        2. Last agent's output
        3. All agent outputs
        """
        # Handle None state
        if final_state is None:
            logger.warning("Final state is None - no state returned from workflow execution")
            return {"error": "Workflow execution returned no state"}

        # Handle non-dict types (shouldn't happen but be defensive)
        if not isinstance(final_state, dict):
            logger.warning(f"Final state is not a dict: {type(final_state)}")
            return {"error": f"Unexpected state type: {type(final_state).__name__}"}

        # Log state for debugging
        logger.debug(f"Extracting output from state with keys: {list(final_state.keys())}")

        # Check for errors
        if final_state.get("error"):
            agent_outputs = final_state.get("agent_outputs", {}) or {}
            logger.info(f"Workflow completed with error: {final_state['error']}")
            return {
                "error": final_state["error"],
                "all_outputs": agent_outputs
            }

        # Check for explicit workflow output
        if final_state.get("workflow_output"):
            logger.info("Returning explicit workflow_output")
            return final_state["workflow_output"]

        # Get agent outputs
        agent_outputs = final_state.get("agent_outputs", {}) or {}

        if agent_outputs:
            logger.info(f"Found outputs from {len(agent_outputs)} agent(s)")

            # Return last agent's output
            last_agent_id = final_state.get("current_agent")
            if last_agent_id and last_agent_id in agent_outputs:
                logger.debug(f"Returning output from last agent: {last_agent_id}")
                return {
                    "result": agent_outputs[last_agent_id],
                    "all_outputs": agent_outputs
                }

            # Return all outputs if we can't determine the last agent
            logger.debug("Returning all agent outputs (no last agent identified)")
            return {"all_outputs": agent_outputs}

        logger.warning("No agent outputs found in final state")
        return {"result": "No output generated"}


class ExecutionManager:
    """
    High-level execution manager

    Handles:
    - Batch executions
    - Scheduled executions (future feature)
    - Execution history and logs
    """

    def __init__(self):
        self.executor = WorkflowExecutor()
        self._execution_history: List[ExecutionStatus] = []

    async def execute_batch(
        self,
        requests: List[ExecuteWorkflowRequest]
    ) -> List[ExecutionStatus]:
        """
        Execute multiple workflows in parallel

        Args:
            requests: List of execution requests

        Returns:
            List of execution statuses
        """
        logger.info(f"Executing batch of {len(requests)} workflows")

        import asyncio

        # Execute all workflows concurrently
        tasks = [
            self.executor.execute(request)
            for request in requests
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed statuses
        statuses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                statuses.append(ExecutionStatus(
                    run_id=f"batch_{i}",
                    workflow_id=requests[i].workflow_id,
                    status="failed",
                    error=str(result),
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow()
                ))
            else:
                statuses.append(result)

        # Store in history
        self._execution_history.extend(statuses)

        logger.info(f"Batch execution completed: {len(statuses)} results")

        return statuses

    def get_execution_history(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionStatus]:
        """
        Get execution history

        Args:
            workflow_id: Filter by workflow ID (optional)
            limit: Maximum number of results

        Returns:
            List of execution statuses
        """
        if workflow_id:
            history = [
                status for status in self._execution_history
                if status.workflow_id == workflow_id
            ]
        else:
            history = self._execution_history

        # Return most recent first
        return sorted(
            history,
            key=lambda x: x.started_at,
            reverse=True
        )[:limit]

    def clear_history(self):
        """Clear execution history"""
        self._execution_history.clear()
        logger.info("Execution history cleared")
