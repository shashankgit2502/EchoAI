"""
Workflow executor.
Executes workflows using LangGraph runtime.
"""
from typing import Dict, Any, Optional
from echolib.utils import new_id


class WorkflowExecutor:
    """
    Workflow execution service.
    Compiles and runs workflows via LangGraph.
    """

    def __init__(self, storage, compiler, agent_registry, guards=None):
        """
        Initialize executor.

        Args:
            storage: WorkflowStorage instance
            compiler: WorkflowCompiler instance
            agent_registry: AgentRegistry instance
            guards: RuntimeGuards instance (optional)
        """
        self.storage = storage
        self.compiler = compiler
        self.agent_registry = agent_registry
        self.guards = guards

    def execute_workflow(
        self,
        workflow_id: str,
        execution_mode: str,
        version: Optional[str] = None,
        input_payload: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow.

        Args:
            workflow_id: Workflow identifier
            execution_mode: "test" or "final"
            version: Version (required for final mode)
            input_payload: Input data for workflow

        Returns:
            Execution result

        Raises:
            RuntimeError: If workflow not in correct state
        """
        if input_payload is None:
            input_payload = {}

        # Load workflow based on mode
        if execution_mode == "draft":
            # Try draft first, fall back to temp for backwards compatibility
            try:
                workflow = self.storage.load_workflow(
                    workflow_id=workflow_id,
                    state="draft"
                )
            except FileNotFoundError:
                # Fallback: try temp folder (for workflows saved before draft-chat feature)
                print(f"Draft not found for {workflow_id}, trying temp folder...")
                workflow = self.storage.load_workflow(
                    workflow_id=workflow_id,
                    state="temp"
                )
            # Draft workflows can be in draft, validated, or testing status (for temp fallback)
            if workflow.get("status") not in ("draft", "validated", "testing"):
                raise RuntimeError("Workflow not in usable state")

        elif execution_mode == "test":
            workflow = self.storage.load_workflow(
                workflow_id=workflow_id,
                state="temp"
            )
            if workflow.get("status") != "testing":
                raise RuntimeError("Workflow not in testing state")

        elif execution_mode == "final":
            if not version:
                raise ValueError("Version required for final execution")
            workflow = self.storage.load_workflow(
                workflow_id=workflow_id,
                state="final",
                version=version
            )
            if workflow.get("status") != "final":
                raise RuntimeError("Workflow not finalized")

        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        # Apply guards if available
        if self.guards:
            self.guards.check_before_execution(workflow)

        # Load agent definitions
        agent_defs = {}
        for agent_id in workflow.get("agents", []):
            try:
                agent = self.agent_registry.get_agent(agent_id)
                if agent is None:
                    # Agent not in cache - try reloading from disk
                    print(f"Warning: Agent '{agent_id}' not in cache, reloading registry...")
                    self.agent_registry._load_all()  # Reload all agents from disk
                    agent = self.agent_registry.get_agent(agent_id)
                    if agent is None:
                        raise RuntimeError(f"Agent '{agent_id}' not found in registry after reload")
                agent_defs[agent_id] = agent
            except FileNotFoundError:
                raise RuntimeError(f"Agent '{agent_id}' not found in registry")

        # Compile workflow to LangGraph
        compiled_graph = self.compiler.compile_to_langgraph(workflow, agent_defs)

        # Execute workflow with LangGraph
        run_id = new_id("run_")

        try:
            # Prepare initial state with workflow and run IDs
            if input_payload is None:
                input_payload = {}

            # Normalize user input: ensure "user_input" key is set
            # Accept common key names from frontend
            # FIX: Handle structured payloads (code, language, etc.) that don't use standard keys
            user_input = (
                input_payload.get("user_input")
                or input_payload.get("message")
                or input_payload.get("question")
                or input_payload.get("input")
                or input_payload.get("task_description")
                or input_payload.get("prompt")
            )

            # FIX: If no standard key found but payload has data, serialize entire payload
            # This handles structured inputs like {"code": "...", "language": "python"}
            if not user_input and input_payload:
                import json
                # Check if there are any meaningful keys in the payload (not just metadata)
                meaningful_keys = {k for k in input_payload.keys()
                                   if k not in ("workflow_id", "run_id", "mode", "version", "context")}
                if meaningful_keys:
                    # Serialize the structured input as JSON string for agents to parse
                    user_input = json.dumps(input_payload, indent=2)
                    print(f"[Executor] Serialized structured payload as user_input: {user_input[:200]}...")

            # Final fallback to empty string
            if not user_input:
                user_input = ""

            # FIXED: Set initial state with original_user_input preserved
            # This ensures all agents throughout the workflow have access
            # to the original user request, not just the first agent
            initial_state = {
                **input_payload,
                "user_input": user_input,
                "original_user_input": user_input,  # CRITICAL: Preserve original
                "task_description": user_input,     # Alias for compatibility
                "messages": [],
                "workflow_id": workflow_id,
                "run_id": run_id
            }

            # Run the compiled graph
            config = {"configurable": {"thread_id": run_id}}
            final_state = compiled_graph.invoke(initial_state, config)

            result = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_mode": execution_mode,
                "output": final_state,
                "messages": final_state.get("messages", [])
            }

        except Exception as e:
            result = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "execution_mode": execution_mode,
                "error": str(e),
                "output": {}
            }

        return result

    def load_for_execution(
        self,
        workflow_id: str,
        mode: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load workflow for execution.

        Args:
            workflow_id: Workflow identifier
            mode: "draft", "test", or "final"
            version: Version (for final mode)

        Returns:
            Workflow definition
        """
        if mode == "draft":
            return self.storage.load_workflow(
                workflow_id=workflow_id,
                state="draft"
            )
        elif mode == "test":
            return self.storage.load_workflow(
                workflow_id=workflow_id,
                state="temp"
            )
        elif mode == "final":
            return self.storage.load_workflow(
                workflow_id=workflow_id,
                state="final",
                version=version
            )
        else:
            raise ValueError("Invalid execution mode")
