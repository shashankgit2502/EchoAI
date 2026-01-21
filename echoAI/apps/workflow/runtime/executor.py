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
        if execution_mode == "test":
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
            # Ensure input_payload is a dict (not None)
            if input_payload is None:
                input_payload = {}

            initial_state = {
                **input_payload,
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
            mode: "test" or "final"
            version: Version (for final mode)

        Returns:
            Workflow definition
        """
        if mode == "test":
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
