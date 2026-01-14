"""
Workflow Compiler - Converts AgentSystemDesign JSON to LangGraph StateGraph
Handles different communication patterns and creates executable workflows
"""
import os  # For Azure deployment environment variables
from typing import Dict, Any, List, Optional, Callable, TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
# For Azure deployment - uncomment the line below
# from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
import operator

from app.schemas.api_models import (
    AgentSystemDesign,
    AgentDefinition,
    WorkflowDefinition,
    WorkflowStep
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm_provider import get_llm_provider

logger = get_logger(__name__)


# ============================================================================
# REDUCER FUNCTIONS for concurrent state updates
# ============================================================================

def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, right values override left"""
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}


def concat_lists(left: List, right: List) -> List:
    """Concatenate two lists"""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


def keep_last(left: Any, right: Any) -> Any:
    """Keep the last non-None value"""
    return right if right is not None else left


def keep_first_non_none(left: Any, right: Any) -> Any:
    """Keep the first non-None value (for immutable fields like workflow_input)"""
    return left if left is not None else right


# ============================================================================
# STATE DEFINITION - Proper TypedDict with Annotated reducers for LangGraph
# ============================================================================

class WorkflowState(TypedDict, total=False):
    """
    State object passed between nodes in the workflow

    Uses TypedDict with Annotated types for proper LangGraph state handling.
    Annotated types define reducer functions for concurrent updates.
    total=False allows optional fields.

    Standard fields:
    - messages: List of conversation messages (appended)
    - current_agent: ID of current agent (last value)
    - agent_outputs: Dict mapping agent_id -> output (merged)
    - workflow_input: Original input to workflow (immutable, keep first)
    - workflow_output: Final output (last value)
    - error: Error message if any (last value)
    - metadata: Additional metadata (merged)
    """
    messages: Annotated[List[Dict[str, Any]], concat_lists]
    current_agent: Annotated[Optional[str], keep_last]
    agent_outputs: Annotated[Dict[str, Any], merge_dicts]
    workflow_input: Annotated[Dict[str, Any], keep_first_non_none]
    workflow_output: Annotated[Optional[Dict[str, Any]], keep_last]
    error: Annotated[Optional[str], keep_last]
    metadata: Annotated[Dict[str, Any], merge_dicts]


# ============================================================================
# WORKFLOW COMPILER
# ============================================================================

class WorkflowCompiler:
    """
    Compiles AgentSystemDesign JSON to executable LangGraph StateGraph

    Handles:
    - Creating agent nodes from agent definitions
    - Building workflow topology from steps
    - Supporting different communication patterns
    - Tool integration (basic, MCP tools added later)
    - Checkpointing for HITL
    """

    def __init__(self):
        self.settings = get_settings()
        self._agent_llms: Dict[str, Any] = {}
        self._agent_tools: Dict[str, List[Callable]] = {}
        self._step_node_mapping: Dict[int, str] = {}  # Maps step index to unique node ID
        self._llm_provider = get_llm_provider()

    def _normalize_openrouter_model(self, model_id: str) -> str:
        if model_id.startswith("openrouter/"):
            return model_id[len("openrouter/"):]
        return model_id

    def compile(
        self,
        agent_system: AgentSystemDesign,
        workflow_name: Optional[str] = None,
        enable_checkpointing: bool = True
    ) -> CompiledStateGraph:
        """
        Compile agent system to executable LangGraph workflow

        Args:
            agent_system: Complete agent system design
            workflow_name: Specific workflow to compile (if None, compiles first workflow)
            enable_checkpointing: Enable checkpointing for HITL

        Returns:
            Compiled StateGraph ready for execution
        """
        logger.info(f"Compiling agent system: {agent_system.system_name}")

        # Select workflow to compile
        workflow = self._select_workflow(agent_system, workflow_name)

        # Determine effective communication pattern
        # Use system-level pattern if hierarchical (has master agent), else use workflow pattern
        has_master = any(agent.is_master for agent in agent_system.agents)
        system_pattern = getattr(agent_system, 'communication_pattern', None)

        if has_master and system_pattern == "hierarchical":
            effective_pattern = "hierarchical"
            logger.info(f"Compiling workflow: {workflow.name} (hierarchical - master agent detected)")
        else:
            effective_pattern = workflow.communication_pattern
            logger.info(f"Compiling workflow: {workflow.name} ({effective_pattern})")

        # Initialize LLMs for all agents
        self._initialize_agents(agent_system.agents)

        # Initialize tools
        self._initialize_tools(agent_system)

        # Build StateGraph
        graph = StateGraph(WorkflowState)

        # For hierarchical, skip step-based node mapping (handled by _build_hierarchical)
        if effective_pattern == "hierarchical":
            self._step_node_mapping = {}
            # Add worker nodes only (master nodes added by _build_hierarchical)
            for agent in agent_system.agents:
                if not agent.is_master:
                    node_func = self._create_agent_node(agent, agent_system)
                    graph.add_node(agent.id, node_func)
                    logger.debug(f"Added worker node: {agent.id}")
        else:
            # Detect duplicate agents in steps and create unique node IDs
            self._step_node_mapping = self._build_step_node_mapping(workflow, agent_system)

            # Add agent nodes (with unique IDs for duplicates)
            added_nodes = set()
            for step_idx, step in enumerate(workflow.steps):
                node_id = self._step_node_mapping.get(step_idx, step.agent_id)
                if node_id not in added_nodes:
                    agent = self._get_agent_by_id(agent_system, step.agent_id)
                    if agent:
                        node_func = self._create_agent_node(agent, agent_system)
                        graph.add_node(node_id, node_func)
                        added_nodes.add(node_id)
                        logger.debug(f"Added node: {node_id}")

        # Build workflow topology based on effective pattern
        if effective_pattern == "sequential":
            self._build_sequential(graph, workflow, agent_system)
        elif effective_pattern == "parallel":
            self._build_parallel(graph, workflow, agent_system)
        elif effective_pattern == "hierarchical":
            self._build_hierarchical(graph, workflow, agent_system)
        elif effective_pattern == "conditional":
            self._build_conditional(graph, workflow, agent_system)
        elif effective_pattern == "graph":
            self._build_graph(graph, workflow, agent_system)
        else:
            raise ValueError(f"Unknown communication pattern: {effective_pattern}")

        # Set entry point
        # For hierarchical: use master_delegate node
        # For others: use first step's node (with mapping for duplicates)
        if hasattr(self, '_hierarchical_entry') and self._hierarchical_entry:
            entry_node = self._hierarchical_entry
            self._hierarchical_entry = None  # Reset for next compile
        elif workflow.steps:
            entry_node = self._get_node_id_for_step(0, workflow.steps[0].agent_id)
        else:
            raise ValueError("No entry point could be determined for workflow")

        graph.set_entry_point(entry_node)
        logger.debug(f"Entry point: {entry_node}")

        # Compile with optional checkpointing
        if enable_checkpointing and self.settings.ENABLE_CHECKPOINTING:
            checkpointer = MemorySaver()
            compiled = graph.compile(checkpointer=checkpointer)
            logger.info("Compiled with checkpointing enabled (HITL support)")
        else:
            compiled = graph.compile()
            logger.info("Compiled without checkpointing")

        return compiled

    def _select_workflow(
        self,
        agent_system: AgentSystemDesign,
        workflow_name: Optional[str]
    ) -> WorkflowDefinition:
        """Select workflow to compile"""
        if workflow_name:
            for workflow in agent_system.workflows:
                if workflow.name == workflow_name:
                    return workflow
            raise ValueError(f"Workflow not found: {workflow_name}")

        # Return first workflow if not specified
        return agent_system.workflows[0]

    def _get_agent_by_id(
        self,
        agent_system: AgentSystemDesign,
        agent_id: str
    ) -> Optional[AgentDefinition]:
        """Get agent definition by ID"""
        for agent in agent_system.agents:
            if agent.id == agent_id:
                return agent
        return None

    def _build_step_node_mapping(
        self,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign
    ) -> Dict[int, str]:
        """
        Build mapping from step index to unique node ID.

        When an agent appears multiple times in a workflow, each occurrence
        gets a unique node ID to prevent graph cycles.

        Example:
            Steps: [coordinator, analyst, writer, coordinator]
            Mapping: {0: "coordinator_0", 3: "coordinator_1"}
            (analyst and writer don't need mapping as they appear once)
        """
        agent_occurrences: Dict[str, List[int]] = {}

        # Count occurrences of each agent
        for step_idx, step in enumerate(workflow.steps):
            agent_id = step.agent_id
            if agent_id not in agent_occurrences:
                agent_occurrences[agent_id] = []
            agent_occurrences[agent_id].append(step_idx)

        # Build mapping for agents that appear more than once
        step_node_mapping: Dict[int, str] = {}
        for agent_id, step_indices in agent_occurrences.items():
            if len(step_indices) > 1:
                # Agent appears multiple times - create unique IDs
                for occurrence_num, step_idx in enumerate(step_indices):
                    unique_node_id = f"{agent_id}_{occurrence_num}"
                    step_node_mapping[step_idx] = unique_node_id
                    logger.debug(f"Mapped step {step_idx} ({agent_id}) to unique node: {unique_node_id}")

        return step_node_mapping

    def _get_node_id_for_step(self, step_idx: int, agent_id: str) -> str:
        """Get the node ID for a given step (handles duplicates)"""
        return self._step_node_mapping.get(step_idx, agent_id)

    def _initialize_agents(self, agents: List[AgentDefinition]):
        """Initialize LLM instances for all agents"""
        logger.info(f"Initializing {len(agents)} agents...")

        for agent in agents:
            llm_config = agent.llm_config
            model_meta = self._llm_provider.get_model_metadata(llm_config.model)
            normalized_model = self._normalize_openrouter_model(llm_config.model)

            # Determine provider from model catalog when possible
            if model_meta and model_meta.provider == "anthropic":
                llm = ChatAnthropic(
                    model=normalized_model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            elif model_meta and model_meta.provider == "openai":
                # For Azure deployment - uncomment this block
                # llm = AzureChatOpenAI(
                #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #     temperature=llm_config.temperature,
                #     max_tokens=llm_config.max_tokens
                # )

                # For local/standard OpenAI - comment this when deploying to Azure
                llm = ChatOpenAI(
                    model=normalized_model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            elif model_meta and model_meta.provider == "openrouter":
                # OpenRouter models from catalog
                # For Azure deployment - uncomment this block
                # llm = AzureChatOpenAI(
                #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #     temperature=llm_config.temperature,
                #     max_tokens=llm_config.max_tokens
                # )

                # For local/OpenRouter - comment this when deploying to Azure
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    openai_api_key=self.settings.OPENROUTER_API_KEY,
                    model=normalized_model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            elif model_meta and model_meta.provider == "onprem":
                # For Azure deployment - uncomment this block
                # llm = AzureChatOpenAI(
                #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #     temperature=llm_config.temperature,
                #     max_tokens=llm_config.max_tokens
                # )

                # For local/on-prem - comment this when deploying to Azure
                llm = ChatOpenAI(
                    base_url=self.settings.ONPREM_BASE_URL if hasattr(self.settings, "ONPREM_BASE_URL") else "http://10.188.100.131:8004/v1",
                    api_key=self.settings.ONPREM_API_KEY if hasattr(self.settings, "ONPREM_API_KEY") else "ollama",
                    model=normalized_model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            elif llm_config.model.startswith("openrouter/"):
                # OpenRouter models not in catalog (fallback)
                # For Azure deployment - uncomment this block
                # llm = AzureChatOpenAI(
                #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #     temperature=llm_config.temperature,
                #     max_tokens=llm_config.max_tokens
                # )

                # For local/OpenRouter - comment this when deploying to Azure
                llm = ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    openai_api_key=self.settings.OPENROUTER_API_KEY,
                    model=normalized_model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                # Default to OpenAI for unknown models
                logger.warning(f"Unknown model type: {llm_config.model}, defaulting to OpenAI")

                # For Azure deployment - uncomment this block
                # llm = AzureChatOpenAI(
                #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #     temperature=llm_config.temperature,
                #     max_tokens=llm_config.max_tokens
                # )

                # For local/standard OpenAI - comment this when deploying to Azure
                llm = ChatOpenAI(
                    model=normalized_model,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )

            self._agent_llms[agent.id] = llm
            logger.debug(f"Initialized {agent.id}: {llm_config.model}")

    def _initialize_tools(self, agent_system: AgentSystemDesign):
        """
        Initialize tools for agents

        For now, creates placeholder tool functions
        MCP integration will be added later
        """
        logger.info(f"Initializing {len(agent_system.tools)} tools...")

        for agent in agent_system.agents:
            agent_tools = []

            for tool_name in agent.tools:
                # Find tool definition
                tool_def = next(
                    (t for t in agent_system.tools if t.name == tool_name),
                    None
                )

                if tool_def:
                    # Create placeholder tool function
                    tool_func = self._create_tool_placeholder(tool_def)
                    agent_tools.append(tool_func)
                    logger.debug(f"Tool '{tool_name}' added to {agent.id}")
                else:
                    logger.warning(f"Tool '{tool_name}' not found for agent {agent.id}")

            self._agent_tools[agent.id] = agent_tools

    def _create_tool_placeholder(self, tool_def) -> Callable:
        """Create placeholder tool function (MCP integration later)"""
        def placeholder_tool(*args, **kwargs):
            logger.warning(f"Tool '{tool_def.name}' called but not yet implemented")
            return {"status": "not_implemented", "tool": tool_def.name}

        placeholder_tool.__name__ = tool_def.name
        return placeholder_tool

    def _create_agent_node(
        self,
        agent: AgentDefinition,
        agent_system: AgentSystemDesign
    ) -> Callable:
        """
        Create a node function for an agent

        The node:
        1. Reads state
        2. Formats input for agent
        3. Calls LLM with system prompt
        4. Processes output
        5. Updates state
        """
        agent_llm = self._agent_llms[agent.id]
        agent_tools = self._agent_tools.get(agent.id, [])

        def agent_node(state: WorkflowState) -> WorkflowState:
            """Execute agent logic"""
            logger.info(f"Executing agent: {agent.id} ({agent.role})")

            # Get workflow input (safely with defaults)
            workflow_input = state.get("workflow_input", {}) or {}

            # Get outputs from previous agents (create new dict to avoid mutation)
            existing_outputs = state.get("agent_outputs", {}) or {}
            agent_outputs = dict(existing_outputs)  # Copy to avoid mutation

            # Build context from previous outputs
            context_messages = []
            if agent_outputs:
                context = "\n\n".join([
                    f"Output from {agent_id}:\n{output}"
                    for agent_id, output in agent_outputs.items()
                ])
                context_messages.append(HumanMessage(content=f"Previous agent outputs:\n{context}"))

            # Add workflow input
            if workflow_input:
                input_str = str(workflow_input)
                context_messages.append(HumanMessage(content=f"Workflow input:\n{input_str}"))

            # Create messages for LLM
            messages = [
                SystemMessage(content=agent.system_prompt)
            ] + context_messages

            # Add current request if no context
            if not context_messages:
                messages.append(HumanMessage(content="Process the input and provide your analysis."))

            try:
                # Call LLM
                response = agent_llm.invoke(messages)

                # Extract content
                if hasattr(response, 'content'):
                    output = response.content
                else:
                    output = str(response)

                logger.info(f"Agent {agent.id} completed successfully")
                if output:
                    logger.debug(f"Output preview: {output[:200]}...")

                # Return ONLY modified fields - reducers will merge with existing state
                # This is critical for parallel execution to work correctly
                return {
                    "agent_outputs": {agent.id: output},  # Only this agent's output
                    "current_agent": agent.id,
                    "error": None
                }

            except Exception as e:
                logger.error(f"Agent {agent.id} failed: {e}")
                # Return only error-related fields
                return {
                    "agent_outputs": {agent.id: f"ERROR: {str(e)}"},
                    "current_agent": agent.id,
                    "error": f"Agent {agent.id} failed: {str(e)}"
                }

        return agent_node

    # ========================================================================
    # TOPOLOGY BUILDERS
    # ========================================================================

    def _build_sequential(
        self,
        graph: StateGraph,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign
    ):
        """
        Build sequential workflow: A -> B -> C -> END

        Uses step-to-node mapping to handle duplicate agents correctly.
        Each step gets a unique node ID if the same agent appears multiple times.
        """
        logger.info("Building sequential topology")

        steps = workflow.steps
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            # Use mapped node IDs to handle duplicate agents
            current_node = self._get_node_id_for_step(i, current_step.agent_id)
            next_node = self._get_node_id_for_step(i + 1, next_step.agent_id)

            graph.add_edge(current_node, next_node)
            logger.debug(f"Edge: {current_node} -> {next_node}")

        # Last step to END
        last_idx = len(steps) - 1
        last_step = steps[last_idx]
        last_node = self._get_node_id_for_step(last_idx, last_step.agent_id)
        graph.add_edge(last_node, END)
        logger.debug(f"Edge: {last_node} -> END")

    def _build_parallel(
        self,
        graph: StateGraph,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign
    ):
        """
        Build parallel workflow with synchronization

        Uses parallel_with field to group steps
        """
        logger.info("Building parallel topology")

        steps = workflow.steps
        processed_indices = set()

        i = 0
        while i < len(steps):
            if i in processed_indices:
                i += 1
                continue

            current_step = steps[i]

            # Check if this step has parallel companions
            if current_step.parallel_with:
                # Find all parallel steps
                parallel_agents = [current_step.agent_id] + current_step.parallel_with

                # All parallel steps point to next sequential step or END
                if i + 1 < len(steps):
                    next_step = steps[i + 1]
                    for agent_id in parallel_agents:
                        graph.add_edge(agent_id, next_step.agent_id)
                        logger.debug(f"Edge: {agent_id} -> {next_step.agent_id}")
                else:
                    for agent_id in parallel_agents:
                        graph.add_edge(agent_id, END)
                        logger.debug(f"Edge: {agent_id} -> END")

                processed_indices.add(i)
            else:
                # Sequential step
                if i + 1 < len(steps):
                    next_step = steps[i + 1]
                    graph.add_edge(current_step.agent_id, next_step.agent_id)
                    logger.debug(f"Edge: {current_step.agent_id} -> {next_step.agent_id}")
                else:
                    graph.add_edge(current_step.agent_id, END)
                    logger.debug(f"Edge: {current_step.agent_id} -> END")

            i += 1

    def _create_hierarchical_master_node(
        self,
        agent: AgentDefinition,
        agent_system: AgentSystemDesign,
        phase: str  # "delegate" or "aggregate"
    ) -> Callable:
        """
        Create a node function for hierarchical master agent phases.

        This is separate from _create_agent_node to avoid affecting other workflows.
        Uses phase-specific output keys and prompts.
        """
        agent_llm = self._agent_llms[agent.id]
        output_key = f"{agent.id}_{phase}"

        # Phase-specific instructions
        if phase == "delegate":
            phase_instruction = (
                "You are starting the workflow. Review the input and prepare context for the worker agents. "
                "Summarize what needs to be done and set expectations."
            )
        else:  # aggregate
            phase_instruction = (
                "All worker agents have completed their tasks. Review ALL their outputs below and "
                "compile a comprehensive FINAL result that integrates all the work done. "
                "This is the final output that will be delivered to the user."
            )

        def master_node(state: WorkflowState) -> WorkflowState:
            """Execute hierarchical master logic"""
            logger.info(f"Executing master ({phase}): {agent.id} ({agent.role})")

            workflow_input = state.get("workflow_input", {}) or {}
            existing_outputs = state.get("agent_outputs", {}) or {}
            agent_outputs = dict(existing_outputs)

            # Build context from previous outputs
            context_messages = []
            if agent_outputs:
                context = "\n\n".join([
                    f"=== Output from {agent_id} ===\n{output}"
                    for agent_id, output in agent_outputs.items()
                ])
                context_messages.append(HumanMessage(content=f"Previous agent outputs:\n{context}"))

            # Add workflow input
            if workflow_input:
                input_str = str(workflow_input)
                context_messages.append(HumanMessage(content=f"Original request:\n{input_str}"))

            # Create messages with phase-specific instruction
            messages = [
                SystemMessage(content=f"{agent.system_prompt}\n\n{phase_instruction}")
            ] + context_messages

            if not context_messages:
                messages.append(HumanMessage(content="Process the input and coordinate the workflow."))

            try:
                response = agent_llm.invoke(messages)
                output = response.content if hasattr(response, 'content') else str(response)

                logger.info(f"Master ({phase}) {agent.id} completed successfully")

                return {
                    "agent_outputs": {output_key: output},
                    "current_agent": output_key,
                    "error": None
                }

            except Exception as e:
                logger.error(f"Master ({phase}) {agent.id} failed: {e}")
                return {
                    "agent_outputs": {output_key: f"ERROR: {str(e)}"},
                    "current_agent": output_key,
                    "error": f"Master ({phase}) {agent.id} failed: {str(e)}"
                }

        return master_node

    def _build_hierarchical(
        self,
        graph: StateGraph,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign
    ):
        """
        Build hierarchical workflow with master coordinator

        Creates two phases for master agent:
        1. master_delegate: Initial coordination, sets up context for workers
        2. master_aggregate: Receives worker outputs, produces final result

        Topology: master_delegate -> worker1 -> worker2 -> ... -> master_aggregate -> END
        """
        logger.info("Building hierarchical topology")

        # Find master agent
        master_agent = next(
            (agent for agent in agent_system.agents if agent.is_master),
            None
        )

        if not master_agent:
            logger.warning("No master agent found for hierarchical pattern, falling back to sequential")
            self._build_sequential(graph, workflow, agent_system)
            return

        # Get worker agents in order (from workflow steps if available, else from agent list)
        worker_agents = self._get_ordered_workers(workflow, agent_system, master_agent.id)

        if not worker_agents:
            logger.warning("No worker agents found, master will run alone")
            graph.add_edge(master_agent.id, END)
            return

        # Create two distinct nodes for master: delegate and aggregate
        master_delegate_id = f"{master_agent.id}_delegate"
        master_aggregate_id = f"{master_agent.id}_aggregate"

        # Add master nodes with phase-specific logic (hierarchical only)
        delegate_node = self._create_hierarchical_master_node(master_agent, agent_system, "delegate")
        aggregate_node = self._create_hierarchical_master_node(master_agent, agent_system, "aggregate")
        graph.add_node(master_delegate_id, delegate_node)
        graph.add_node(master_aggregate_id, aggregate_node)
        logger.debug(f"Added hierarchical master nodes: {master_delegate_id}, {master_aggregate_id}")

        # Update entry point to delegation phase
        self._hierarchical_entry = master_delegate_id

        # Build edges dynamically based on workflow step definitions
        # Respects parallel_with relationships defined by the designer LLM
        self._build_hierarchical_edges(
            graph, workflow, master_delegate_id, master_aggregate_id,
            worker_agents, master_agent.id
        )

        # master_aggregate -> END
        graph.add_edge(master_aggregate_id, END)
        logger.debug(f"Edge: {master_aggregate_id} -> END")

    def _build_hierarchical_edges(
        self,
        graph: StateGraph,
        workflow: WorkflowDefinition,
        delegate_id: str,
        aggregate_id: str,
        worker_agents: List[str],
        master_id: str
    ):
        """
        Build hierarchical edges dynamically based on workflow step definitions.

        Supports:
        - Hierarchical + Sequential: workers run one after another
        - Hierarchical + Parallel: workers with parallel_with run concurrently
        - Hierarchical + Hybrid: mix of sequential and parallel groups

        The designer LLM defines the topology via parallel_with in steps.
        """
        # Get worker steps (excluding master)
        worker_steps = [s for s in workflow.steps if s.agent_id != master_id]

        if not worker_steps:
            # No steps defined - use agent order, all sequential
            logger.info("No worker steps defined, using sequential agent order")
            self._build_hierarchical_sequential(graph, delegate_id, aggregate_id, worker_agents)
            return

        # Analyze parallel_with relationships to build execution groups
        execution_groups = self._analyze_parallel_groups(worker_steps, worker_agents)

        if not execution_groups:
            # Fallback to sequential if analysis fails
            logger.warning("Could not analyze parallel groups, falling back to sequential")
            self._build_hierarchical_sequential(graph, delegate_id, aggregate_id, worker_agents)
            return

        # Build edges based on execution groups
        logger.info(f"Building hierarchical topology with {len(execution_groups)} execution group(s)")

        prev_group = None
        for group_idx, group in enumerate(execution_groups):
            group_agents = group["agents"]
            is_parallel = group["parallel"]

            if group_idx == 0:
                # First group: delegate -> group
                for agent_id in group_agents:
                    graph.add_edge(delegate_id, agent_id)
                    logger.debug(f"Edge: {delegate_id} -> {agent_id}")
            else:
                # Connect from previous group
                prev_agents = prev_group["agents"]
                for prev_agent in prev_agents:
                    for curr_agent in group_agents:
                        graph.add_edge(prev_agent, curr_agent)
                        logger.debug(f"Edge: {prev_agent} -> {curr_agent}")

            prev_group = group

        # Last group -> aggregate
        if prev_group:
            for agent_id in prev_group["agents"]:
                graph.add_edge(agent_id, aggregate_id)
                logger.debug(f"Edge: {agent_id} -> {aggregate_id}")

        # Log topology summary
        topology_desc = self._describe_topology(execution_groups, delegate_id, aggregate_id)
        logger.info(f"Hierarchical topology: {topology_desc}")

    def _analyze_parallel_groups(
        self,
        worker_steps: List,
        worker_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze workflow steps to identify parallel execution groups.

        Returns list of groups:
        [
            {"agents": ["agent1", "agent2"], "parallel": True},
            {"agents": ["agent3"], "parallel": False},
            ...
        ]
        """
        groups = []
        processed = set()

        for step in worker_steps:
            if step.agent_id in processed:
                continue

            if step.parallel_with:
                # This step runs in parallel with others
                parallel_agents = [step.agent_id] + [
                    a for a in step.parallel_with
                    if a in worker_agents and a not in processed
                ]
                groups.append({
                    "agents": parallel_agents,
                    "parallel": True
                })
                processed.update(parallel_agents)
            else:
                # Sequential step
                if step.agent_id in worker_agents:
                    groups.append({
                        "agents": [step.agent_id],
                        "parallel": False
                    })
                    processed.add(step.agent_id)

        # Add any remaining workers not in steps (fallback)
        remaining = [a for a in worker_agents if a not in processed]
        if remaining:
            groups.append({
                "agents": remaining,
                "parallel": len(remaining) > 1
            })

        return groups

    def _build_hierarchical_sequential(
        self,
        graph: StateGraph,
        delegate_id: str,
        aggregate_id: str,
        worker_agents: List[str]
    ):
        """Build simple sequential hierarchical topology (fallback)."""
        logger.info("Building hierarchical sequential topology")

        # delegate -> first worker
        graph.add_edge(delegate_id, worker_agents[0])
        logger.debug(f"Edge: {delegate_id} -> {worker_agents[0]}")

        # Chain workers
        for i in range(len(worker_agents) - 1):
            graph.add_edge(worker_agents[i], worker_agents[i + 1])
            logger.debug(f"Edge: {worker_agents[i]} -> {worker_agents[i + 1]}")

        # Last worker -> aggregate
        graph.add_edge(worker_agents[-1], aggregate_id)
        logger.debug(f"Edge: {worker_agents[-1]} -> {aggregate_id}")

        logger.info(f"Sequential: {delegate_id} -> {' -> '.join(worker_agents)} -> {aggregate_id}")

    def _describe_topology(
        self,
        groups: List[Dict[str, Any]],
        delegate_id: str,
        aggregate_id: str
    ) -> str:
        """Generate human-readable topology description."""
        parts = [delegate_id]

        for group in groups:
            if group["parallel"] and len(group["agents"]) > 1:
                parts.append(f"[{' || '.join(group['agents'])}]")
            else:
                parts.extend(group["agents"])

        parts.append(aggregate_id)
        return " -> ".join(parts)

    def _get_ordered_workers(
        self,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign,
        master_id: str
    ) -> List[str]:
        """
        Get worker agents in execution order.

        Prefers order from workflow steps if available,
        otherwise uses agent definition order.
        """
        # Try to get order from workflow steps (excluding master)
        if workflow.steps:
            step_agents = []
            for step in workflow.steps:
                if step.agent_id != master_id and step.agent_id not in step_agents:
                    step_agents.append(step.agent_id)
            if step_agents:
                return step_agents

        # Fallback: use agent definition order
        return [agent.id for agent in agent_system.agents if agent.id != master_id]

    def _build_conditional(
        self,
        graph: StateGraph,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign
    ):
        """
        Build conditional workflow with branching logic

        Uses 'condition' field in steps to determine routing
        """
        logger.info("Building conditional topology")

        steps = workflow.steps

        for i, step in enumerate(steps):
            if step.condition:
                # Create conditional routing function
                def make_router(step_condition: str, true_agent: str, false_agent: str):
                    def router(state: WorkflowState) -> str:
                        """Route based on condition evaluation"""
                        # Simple condition evaluation (can be enhanced)
                        # For now, check if condition string is in previous output
                        agent_outputs = state.get("agent_outputs", {})
                        last_output = list(agent_outputs.values())[-1] if agent_outputs else ""

                        if step_condition.lower() in last_output.lower():
                            logger.debug(f"Condition '{step_condition}' met, routing to {true_agent}")
                            return true_agent
                        else:
                            logger.debug(f"Condition '{step_condition}' not met, routing to {false_agent}")
                            return false_agent

                    return router

                # Determine true and false branches
                true_agent = step.agent_id
                false_agent = steps[i + 1].agent_id if i + 1 < len(steps) else END

                if i > 0:
                    prev_agent = steps[i - 1].agent_id
                    router_func = make_router(step.condition, true_agent, false_agent)
                    graph.add_conditional_edges(
                        prev_agent,
                        router_func,
                        {true_agent: true_agent, false_agent: false_agent}
                    )
                    logger.debug(f"Conditional edge from {prev_agent}")
            else:
                # Regular sequential edge
                if i + 1 < len(steps):
                    next_step = steps[i + 1]
                    graph.add_edge(step.agent_id, next_step.agent_id)
                else:
                    graph.add_edge(step.agent_id, END)

    def _build_graph(
        self,
        graph: StateGraph,
        workflow: WorkflowDefinition,
        agent_system: AgentSystemDesign
    ):
        """
        Build arbitrary graph topology

        Uses explicit connections defined in workflow steps
        """
        logger.info("Building graph topology")

        # For graph pattern, assume steps define explicit connections
        # Each step can specify next steps via metadata or we infer from order

        steps = workflow.steps

        # Build explicit edges from step definitions
        for i, step in enumerate(steps):
            # Check if step has explicit next steps (via inputs or metadata)
            if "next_steps" in step.inputs:
                next_agent_ids = step.inputs["next_steps"]
                for next_id in next_agent_ids:
                    graph.add_edge(step.agent_id, next_id)
                    logger.debug(f"Edge: {step.agent_id} -> {next_id}")
            else:
                # Default: connect to next step or END
                if i + 1 < len(steps):
                    graph.add_edge(step.agent_id, steps[i + 1].agent_id)
                else:
                    graph.add_edge(step.agent_id, END)

        logger.info("Graph topology built (custom edges)")
