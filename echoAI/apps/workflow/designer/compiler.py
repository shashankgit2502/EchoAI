"""
Workflow compiler.
Compiles workflow JSON definitions into executable LangGraph.

ARCHITECTURAL RULES (STRICTLY ENFORCED):
=========================================
1. LangGraph OWNS: Workflow topology, execution order, branching, merging, state
2. Every workflow is MATERIALIZED AS: LangGraph StateGraph
3. CrewAI is INVOKED: Inside LangGraph nodes (never controls graph)
4. CrewAI HANDLES: Agent collaboration within nodes
5. CrewAI NEVER: Controls graph traversal or state transitions
6. Workflow type: Inferred by designer, compiled here into LangGraph structure
"""
from typing import Dict, Any, TypedDict, List, Annotated
import operator
import os
import logging

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """
    Compiles workflow JSON to executable LangGraph.

    This compiler creates LangGraph StateGraph structures where:
    - Graph topology is defined by this compiler (NOT by agents or CrewAI)
    - Nodes can execute agents via CrewAI (but CrewAI doesn't decide flow)
    - State management is controlled by LangGraph
    """

    def __init__(self, use_crewai: bool = True):
        """
        Initialize compiler.

        Args:
            use_crewai: If True, use CrewAI for agent execution (recommended).
                       If False, use direct LLM calls (legacy mode).
        """
        self._compiled_cache = {}
        self._use_crewai = use_crewai
        self._crewai_adapter = None

        if use_crewai:
            try:
                from ..crewai_adapter import CrewAIAdapter
                self._crewai_adapter = CrewAIAdapter()
                logger.info("CrewAI adapter initialized successfully")
            except ImportError:
                logger.warning("CrewAI not available, falling back to direct LLM execution")
                self._use_crewai = False

    def compile_to_langgraph(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> Any:
        """
        Compile workflow JSON to executable LangGraph.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            Compiled LangGraph instance (runnable)
        """
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError:
            raise ImportError(
                "LangGraph not installed. Run: pip install langgraph langchain-core"
            )

        execution_model = workflow.get("execution_model", "sequential")

        # Create state schema
        WorkflowState = self._create_state_class(workflow, agent_registry)

        # Build graph based on execution model
        # IMPORTANT: All compilation methods build LangGraph structures
        # CrewAI is ONLY used inside node functions, never for graph topology
        if execution_model == "sequential":
            return self._compile_sequential(workflow, agent_registry, WorkflowState)
        elif execution_model == "parallel":
            return self._compile_parallel(workflow, agent_registry, WorkflowState)
        elif execution_model == "hierarchical":
            return self._compile_hierarchical(workflow, agent_registry, WorkflowState)
        elif execution_model == "hybrid":
            return self._compile_hybrid(workflow, agent_registry, WorkflowState)
        else:
            logger.warning(f"Unknown execution model '{execution_model}', defaulting to sequential")
            return self._compile_sequential(workflow, agent_registry, WorkflowState)

    def _create_state_class(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> type:
        """
        Create TypedDict state class for workflow.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            TypedDict class for state
        """
        # Collect all state keys from agents
        state_keys = set()

        for agent_id in workflow.get("agents", []):
            agent = agent_registry.get(agent_id, {})
            state_keys.update(agent.get("input_schema", []))
            state_keys.update(agent.get("output_schema", []))

        # Add workflow-level state keys
        state_keys.update(workflow.get("state_schema", {}).keys())

        # Create TypedDict dynamically
        # Use plain Any for data fields (overwrite semantics)
        # Only messages uses operator.add (accumulate history)
        fields = {key: Any for key in state_keys}

        # Add standard workflow fields
        fields["user_input"] = Any
        fields["task_description"] = Any
        fields["crew_result"] = Any

        # FIXED: Add original_user_input to preserve user query throughout workflow
        # This ensures all agents have access to the original request
        fields["original_user_input"] = Any

        # Parallel execution fields
        fields["parallel_output"] = Any
        fields["individual_outputs"] = Any
        fields["hierarchical_output"] = Any

        # Messages field accumulates across nodes
        fields["messages"] = Annotated[List[Dict[str, Any]], operator.add]

        WorkflowState = TypedDict("WorkflowState", fields)
        return WorkflowState

    def _compile_sequential(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile sequential workflow.

        ARCHITECTURE:
        - LangGraph creates linear chain of nodes: A → B → C → END
        - Each node can use CrewAI for agent execution
        - LangGraph controls the sequence (CrewAI doesn't decide "what's next")
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling sequential workflow...")

        graph = StateGraph(WorkflowState)

        agents = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        # Add agent nodes (CrewAI used inside if enabled)
        for agent_id in agents:
            agent = agent_registry.get(agent_id, {})
            node_func = self._create_agent_node(agent_id, agent)
            graph.add_node(agent_id, node_func)

        # Add edges based on connections (LangGraph controls sequence)
        for i, connection in enumerate(connections):
            from_agent = connection.get("from")
            to_agent = connection.get("to")

            if i == 0:
                # First connection - set entry point
                graph.set_entry_point(from_agent)

            graph.add_edge(from_agent, to_agent)

        # Set finish point
        if agents:
            graph.add_edge(agents[-1], END)

        # Compile with memory
        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        logger.info("Sequential workflow compiled successfully")
        return compiled

    def _compile_parallel(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile parallel workflow.

        ARCHITECTURE (FIXED):
        - LangGraph creates: coordinator → parallel_crew_node → END
        - SINGLE parallel_crew_node executes ALL agents via CrewAI Crew
        - CrewAI handles true parallel agent execution INSIDE the node
        - Results are aggregated within the Crew and returned to LangGraph

        This is the correct architecture because:
        1. LangGraph's invoke() processes nodes sequentially
        2. CrewAI's Crew can execute multiple agents with true parallelism
        3. Using a single Crew node allows agents to share context
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling parallel workflow with CrewAI parallel execution...")

        graph = StateGraph(WorkflowState)

        agents = workflow.get("agents", [])
        agent_configs = [agent_registry.get(aid, {}) for aid in agents if aid in agent_registry]

        # Preserve original user input in coordinator
        def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Coordinator preserves original user input and prepares for parallel execution.
            """
            logger.info(f"Coordinator preparing {len(agents)} agents for parallel execution")
            # Preserve original user input for all downstream agents
            original_input = state.get("user_input") or state.get("message") or state.get("task_description") or ""
            return {
                "original_user_input": original_input,
                "messages": [{
                    "node": "coordinator",
                    "action": "preparing_parallel_execution",
                    "agent_count": len(agents)
                }]
            }

        graph.add_node("coordinator", coordinator)
        graph.set_entry_point("coordinator")

        if self._use_crewai and self._crewai_adapter and agent_configs:
            # USE CREWAI FOR TRUE PARALLEL EXECUTION
            # Create SINGLE node that runs ALL parallel agents in one CrewAI Crew
            logger.info(f"Using CrewAI parallel Crew with {len(agent_configs)} agents")

            aggregation_strategy = workflow.get("aggregation_strategy", "combine")
            parallel_node_func = self._crewai_adapter.create_parallel_crew_node(
                agent_configs=agent_configs,
                aggregation_strategy=aggregation_strategy
            )

            # Single parallel execution node
            graph.add_node("parallel_execution", parallel_node_func)
            graph.add_edge("coordinator", "parallel_execution")
            graph.add_edge("parallel_execution", END)

        else:
            # FALLBACK: Legacy individual node execution (not true parallel)
            logger.warning("CrewAI not available, falling back to sequential-like parallel execution")

            from ..crewai_adapter import create_crewai_merge_node

            # Add individual agent nodes
            for agent_id in agents:
                agent = agent_registry.get(agent_id, {})
                node_func = self._create_agent_node(agent_id, agent)
                graph.add_node(agent_id, node_func)
                graph.add_edge("coordinator", agent_id)

            # Add aggregator
            def aggregator(state: Dict[str, Any]) -> Dict[str, Any]:
                """Aggregate results from parallel agents."""
                logger.info("Aggregating results from parallel execution")
                return state

            graph.add_node("aggregator", aggregator)

            for agent_id in agents:
                graph.add_edge(agent_id, "aggregator")

            graph.add_edge("aggregator", END)

        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        logger.info("Parallel workflow compiled successfully")
        return compiled

    def _compile_hierarchical(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile hierarchical workflow.

        ARCHITECTURE with CrewAI:
        - LangGraph creates a SINGLE node for hierarchical coordination
        - Inside that node, CrewAI Manager delegates to Workers
        - CrewAI handles the delegation logic WITHIN the node
        - LangGraph controls when/if that node executes

        ARCHITECTURE without CrewAI (legacy):
        - LangGraph creates master + sub-agent nodes with bidirectional edges
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling hierarchical workflow...")

        graph = StateGraph(WorkflowState)

        hierarchy = workflow.get("hierarchy", {})
        master_agent_id = hierarchy.get("master_agent")
        sub_agent_ids = hierarchy.get("delegation_order", [])
        delegation_strategy = hierarchy.get("delegation_strategy", "dynamic")

        if not master_agent_id:
            raise ValueError("Hierarchical workflow requires a master agent")

        # Get agent configurations
        master_config = agent_registry.get(master_agent_id, {})
        sub_configs = [agent_registry.get(aid, {}) for aid in sub_agent_ids if aid in agent_registry]

        if self._use_crewai and self._crewai_adapter and sub_configs:
            # USE CREWAI FOR HIERARCHICAL DELEGATION
            # Create single LangGraph node that contains CrewAI hierarchical crew
            logger.info(f"Using CrewAI for hierarchical delegation: {len(sub_configs)} workers")

            hierarchical_node_func = self._crewai_adapter.create_hierarchical_crew_node(
                master_agent_config=master_config,
                sub_agent_configs=sub_configs,
                delegation_strategy=delegation_strategy
            )

            # LangGraph graph is simple: Entry → Hierarchical Node → END
            # All complexity is INSIDE the node via CrewAI
            graph.add_node("hierarchical_master", hierarchical_node_func)
            graph.set_entry_point("hierarchical_master")
            graph.add_edge("hierarchical_master", END)

        else:
            # LEGACY MODE: Direct LLM calls with LangGraph edges
            logger.warning("CrewAI not available, using legacy hierarchical execution")

            # Add master agent node
            master_func = self._create_agent_node(master_agent_id, master_config)
            graph.add_node(master_agent_id, master_func)
            graph.set_entry_point(master_agent_id)

            # Add sub-agent nodes with bidirectional edges
            for agent_id in sub_agent_ids:
                if agent_id in agent_registry:
                    agent = agent_registry.get(agent_id)
                    node_func = self._create_agent_node(agent_id, agent)
                    graph.add_node(agent_id, node_func)
                    graph.add_edge(master_agent_id, agent_id)
                    graph.add_edge(agent_id, master_agent_id)

            graph.add_edge(master_agent_id, END)

        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        logger.info("Hierarchical workflow compiled successfully")
        return compiled

    def _compile_hybrid(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile hybrid workflow (parallel + sequential patterns).

        ARCHITECTURE (FIXED):
         coordinator → parallel_crew_node → merge → [sequential agents] → END

        The parallel section uses a SINGLE CrewAI Crew node for true parallel execution.
        Sequential sections use individual agent nodes.

        Args:
            workflow: Workflow definition with topology
            agent_registry: Agent definitions
            WorkflowState: State type

        Returns:
            Compiled LangGraph
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling hybrid workflow with CrewAI parallel sections...")

        graph = StateGraph(WorkflowState)

        # Extract topology from workflow
        topology = workflow.get("topology", {})
        parallel_groups = topology.get("parallel_groups", [])
        sequential_chains = topology.get("sequential_chains", [])

        if not parallel_groups and not sequential_chains:
            # No topology specified - try to infer from connections
            logger.warning("No topology specified for hybrid workflow, attempting to infer")
            return self._compile_hybrid_from_connections(workflow, agent_registry, WorkflowState, graph)

        # Create coordinator node that preserves original user input
        def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
            """Coordinator preserves user input and prepares hybrid execution."""
            original_input = state.get("user_input") or state.get("message") or state.get("task_description") or ""
            logger.info(f"Hybrid coordinator preparing execution with {len(parallel_groups)} parallel groups")
            return {
                "original_user_input": original_input,
                "messages": [{
                    "node": "coordinator",
                    "action": "preparing_hybrid_execution",
                    "parallel_groups": len(parallel_groups),
                    "sequential_chains": len(sequential_chains)
                }]
            }

        graph.add_node("coordinator", coordinator)
        graph.set_entry_point("coordinator")

        # Collect all parallel agent configs for CrewAI parallel execution
        all_parallel_agent_configs = []
        for group in parallel_groups:
            agent_ids = group.get("agents", [])
            for agent_id in agent_ids:
                agent = agent_registry.get(agent_id, {})
                if agent:
                    all_parallel_agent_configs.append(agent)

        # Use CrewAI for parallel section if available and there are parallel agents
        if self._use_crewai and self._crewai_adapter and all_parallel_agent_configs:
            # CREATE SINGLE PARALLEL CREW NODE for all parallel agents
            logger.info(f"Using CrewAI parallel Crew for {len(all_parallel_agent_configs)} parallel agents")

            merge_strategy = parallel_groups[0].get("merge_strategy", "combine") if parallel_groups else "combine"
            parallel_node_func = self._crewai_adapter.create_parallel_crew_node(
                agent_configs=all_parallel_agent_configs,
                aggregation_strategy=merge_strategy
            )

            # Single parallel execution node
            graph.add_node("parallel_execution", parallel_node_func)
            graph.add_edge("coordinator", "parallel_execution")

            # Merge node processes the parallel output
            def merge_func(state: Dict[str, Any]) -> Dict[str, Any]:
                """Merge processes parallel results for sequential chain."""
                logger.info("Merge node processing parallel execution results")
                # parallel_output and crew_result are already set by parallel_execution
                return {
                    "messages": [{
                        "node": "merge",
                        "action": "parallel_results_merged",
                        "individual_output_count": len(state.get("individual_outputs", []))
                    }]
                }

            graph.add_node("merge", merge_func)
            graph.add_edge("parallel_execution", "merge")

        else:
            # FALLBACK: Legacy individual node execution (not true parallel)
            logger.warning("CrewAI not available, falling back to individual parallel nodes")

            for group in parallel_groups:
                agent_ids = group.get("agents", [])
                for agent_id in agent_ids:
                    agent = agent_registry.get(agent_id, {})
                    node_func = self._create_agent_node(agent_id, agent)
                    graph.add_node(agent_id, node_func)
                    graph.add_edge("coordinator", agent_id)

            # Create merge node
            from ..crewai_adapter import create_crewai_merge_node
            merge_strategy = parallel_groups[0].get("merge_strategy", "combine") if parallel_groups else "combine"
            merge_func = create_crewai_merge_node(all_parallel_agent_configs, merge_strategy)
            graph.add_node("merge", merge_func)

            # Connect parallel agents to merge
            for group in parallel_groups:
                agent_ids = group.get("agents", [])
                for agent_id in agent_ids:
                    graph.add_edge(agent_id, "merge")

        # Add sequential chain after merge (LangGraph edges)
        prev_node = "merge"
        for chain in sequential_chains:
            agent_ids = chain.get("agents", [])
            for agent_id in agent_ids:
                agent = agent_registry.get(agent_id, {})
                node_func = self._create_agent_node(agent_id, agent)
                graph.add_node(agent_id, node_func)
                # LangGraph creates sequential edge
                graph.add_edge(prev_node, agent_id)
                prev_node = agent_id

        # Final edge to END (LangGraph topology)
        graph.add_edge(prev_node, END)

        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)

        logger.info("Hybrid workflow compiled successfully")
        return compiled

    def _compile_hybrid_from_connections(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type,
        graph: Any
    ) -> Any:
        """
        Build hybrid workflow by properly traversing the entire connection graph.

        This method handles ANY workflow structure with:
        - Sequential sections (linear chains)
        - Parallel sections (fan-out from single node to multiple targets)
        - Merge points (fan-in from multiple sources to single target)
        - Multiple parallel/merge sections in the same workflow
        - HITL nodes in any position
        - Conditional/Router branching (routes to ONE target, not parallel)

        Algorithm:
        1. Build adjacency lists from connections (outgoing and incoming edges)
        2. Identify node types: conditional nodes, parallel sources, merge targets, entry points, terminals
        3. Use BFS traversal starting from entry point
        4. Handle CONDITIONAL nodes using LangGraph add_conditional_edges (routes to ONE branch)
        5. Handle parallel sources by creating CrewAI parallel Crew nodes (executes ALL branches)
        6. Handle merge targets by connecting parallel outputs
        7. Handle sequential nodes individually
        8. Connect terminal nodes to LangGraph END

        IMPORTANT:
        - Conditional/Router nodes are NOT parallel sources - they route to ONE target based on a condition
        - This function handles Start/End pseudo-nodes from canvas and properly filters them
        """
        from langgraph.graph import END
        from langgraph.checkpoint.memory import MemorySaver
        from collections import deque

        logger.info("Building hybrid workflow by traversing connection graph...")

        agents = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        if not agents:
            raise ValueError("No agents defined for hybrid workflow")

        # =====================================================================
        # STEP 1: Normalize connections - handle Start/End pseudo-nodes
        # =====================================================================
        normalized_connections = []
        canvas_start_targets = []  # Nodes that Start connects to

        for conn in connections:
            from_node = conn.get("from", "")
            to_node = conn.get("to", "")

            # Convert to string for comparison (handle integer IDs from canvas)
            from_node_str = str(from_node) if from_node else ""
            to_node_str = str(to_node) if to_node else ""

            # Detect Start pseudo-node (not in agents list, connects TO agents)
            if from_node_str and from_node_str not in agents and to_node_str in agents:
                if "start" in from_node_str.lower():
                    logger.debug(f"Detected Start pseudo-node: {from_node}")
                    canvas_start_targets.append(to_node_str)
                    continue

            # Detect End pseudo-node (not in agents list, receives FROM agents)
            if to_node_str and to_node_str not in agents and from_node_str in agents:
                if "end" in to_node_str.lower():
                    logger.debug(f"Detected End pseudo-node: {to_node}")
                    continue

            # Only include connections between actual agents
            if from_node_str in agents and to_node_str in agents:
                normalized_connections.append({
                    "from": from_node_str,
                    "to": to_node_str
                })

        # Use normalized connections for analysis
        connections = normalized_connections
        logger.info(f"Using {len(connections)} agent-to-agent connections")

        # =====================================================================
        # STEP 2: Build adjacency lists
        # =====================================================================
        outgoing = {agent: [] for agent in agents}  # node -> [targets]
        incoming = {agent: [] for agent in agents}  # node -> [sources]

        for conn in connections:
            from_node = conn.get("from")
            to_node = conn.get("to")
            if from_node in outgoing and to_node in incoming:
                outgoing[from_node].append(to_node)
                incoming[to_node].append(from_node)

        # =====================================================================
        # STEP 3: Identify node types
        # =====================================================================
        # Build node_types dictionary for filtering
        node_types = {}
        for agent_id in agents:
            agent_config = agent_registry.get(agent_id, {})
            # Check both metadata.node_type and top-level type field
            node_type = agent_config.get("metadata", {}).get("node_type", "")
            if not node_type:
                node_type = agent_config.get("type", "")
            node_types[agent_id] = node_type

        # Conditional/Router nodes: nodes with multiple outgoing edges that should BRANCH (not parallel)
        # These nodes route to ONE target based on a condition, not all targets simultaneously
        conditional_nodes = {
            a for a in agents
            if len(outgoing[a]) > 1
            and node_types.get(a) in ("Conditional", "Router", "conditional", "router")
        }

        # Parallel sources: nodes with multiple outgoing edges EXCLUDING Conditional/Router
        # These nodes truly fan out to execute multiple branches in parallel
        parallel_sources = {
            a for a in agents
            if len(outgoing[a]) > 1
            and a not in conditional_nodes
        }

        # Merge targets: nodes with multiple incoming edges
        merge_targets = {a for a in agents if len(incoming[a]) > 1}

        logger.info(f"  Conditional nodes ({len(conditional_nodes)}): {conditional_nodes}")

        # Entry point: node with no incoming edges (or first in canvas_start_targets)
        if canvas_start_targets:
            entry_point = canvas_start_targets[0]
        else:
            entry_candidates = [a for a in agents if not incoming[a]]
            entry_point = entry_candidates[0] if entry_candidates else agents[0]

        # Terminal nodes: nodes with no outgoing edges
        terminal_nodes = {a for a in agents if not outgoing[a]}

        logger.info(f"Graph analysis complete:")
        logger.info(f"  Entry point: {entry_point}")
        logger.info(f"  Parallel sources ({len(parallel_sources)}): {parallel_sources}")
        logger.info(f"  Merge targets ({len(merge_targets)}): {merge_targets}")
        logger.info(f"  Terminal nodes ({len(terminal_nodes)}): {terminal_nodes}")

        # =====================================================================
        # STEP 4: Create coordinator node (preserves user input)
        # =====================================================================
        def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
            """Coordinator preserves user input and prepares for hybrid execution."""
            original_input = (
                state.get("user_input")
                or state.get("message")
                or state.get("task_description")
                or ""
            )
            logger.info(f"Hybrid coordinator starting execution with {len(agents)} agents")
            return {
                "original_user_input": original_input,
                "messages": [{
                    "node": "coordinator",
                    "action": "hybrid_execution_start",
                    "total_agents": len(agents),
                    "parallel_sections": len(parallel_sources),
                    "merge_points": len(merge_targets)
                }]
            }

        graph.add_node("coordinator", coordinator)
        graph.set_entry_point("coordinator")

        # =====================================================================
        # STEP 5: BFS traversal to build the full graph
        # =====================================================================
        visited = set()  # Agents that have been processed
        nodes_created = set()  # LangGraph nodes that have been created
        parallel_crew_counter = 0  # Counter for unique parallel crew names
        conditional_targets = set()  # Track nodes reached via conditional edges

        # Track which parallel crew a merge target should connect from
        merge_target_to_parallel_crew = {}

        # Queue: (agent_id, previous_langgraph_node)
        queue = deque([(entry_point, "coordinator")])

        while queue:
            current, prev_lg_node = queue.popleft()

            if current is None or current in visited:
                continue

            visited.add(current)

            # Skip if this node is a target of a parallel section (handled by Crew)
            # But only if it's NOT also a merge target (merge targets need individual nodes)
            # IMPORTANT: Do NOT skip nodes that are targets of conditional nodes
            is_parallel_target = any(
                current in outgoing.get(ps, [])
                for ps in parallel_sources
            )

            # Check if this node is a target of a conditional node (should NOT be skipped)
            is_conditional_target = any(
                current in outgoing.get(cn, [])
                for cn in conditional_nodes
            )

            if is_parallel_target and current not in merge_targets and current not in parallel_sources and not is_conditional_target:
                # This node is handled by a parallel Crew, skip individual creation
                logger.debug(f"Skipping {current} - handled by parallel Crew")
                continue

            # Get agent config
            agent_config = agent_registry.get(current, {})

            # Check if this is a HITL node
            node_type = agent_config.get("metadata", {}).get("node_type", "")
            is_hitl = node_type == "HITL" or agent_config.get("type") == "HITL"

            # =====================================================================
            # CASE A0: Current node is a CONDITIONAL/ROUTER (branches to ONE target)
            # =====================================================================
            # IMPORTANT: Conditional nodes must NOT execute branches in parallel.
            # They evaluate a condition and route to exactly ONE branch.
            if current in conditional_nodes:
                logger.info(f"Processing conditional node: {current}")

                # Create the conditional node itself
                if current not in nodes_created:
                    conditional_func = self._create_conditional_node(current, agent_config)
                    graph.add_node(current, conditional_func)
                    nodes_created.add(current)
                    graph.add_edge(prev_lg_node, current)
                    logger.info(f"Created conditional node: {current}")

                # Get branch targets from config
                branch_targets = outgoing[current]
                branches_config = agent_config.get("config", {}).get("branches", [])

                # Create routing function for LangGraph conditional edges
                routing_func = self._create_conditional_routing_function(
                    current, agent_config, branch_targets, branches_config
                )

                # Build path_map: maps routing function return values to target nodes
                # The routing function returns the target node ID directly
                path_map = {target: target for target in branch_targets}

                # Add conditional edges using LangGraph's native routing
                graph.add_conditional_edges(
                    current,
                    routing_func,
                    path_map
                )
                logger.info(f"Added conditional edges from {current} to targets: {branch_targets}")

                # Mark all branch targets as conditional targets (they already have conditional edges)
                for target in branch_targets:
                    conditional_targets.add(target)

                # Queue all branch targets for individual processing
                # Each branch is processed as a separate sequential chain
                for target in branch_targets:
                    if target not in visited:
                        queue.append((target, current))

            # =====================================================================
            # CASE A: Current node is a PARALLEL SOURCE (fans out to multiple targets)
            # =====================================================================
            elif current in parallel_sources:
                logger.info(f"Processing parallel source: {current}")

                # First, create the parallel source node itself (if it's not just routing)
                # Check if this is a pure routing node (like Conditional) or an agent
                agent_role = agent_config.get("role", "")
                is_pure_router = agent_config.get("type") in ("Conditional", "Router")

                if not is_pure_router and current not in nodes_created:
                    # Create the source agent node
                    source_func = self._create_agent_node(current, agent_config)
                    graph.add_node(current, source_func)
                    nodes_created.add(current)
                    graph.add_edge(prev_lg_node, current)
                    prev_lg_node = current
                    logger.info(f"Created agent node for parallel source: {current}")
                elif is_pure_router and current not in nodes_created:
                    # For pure routers, still create the node
                    router_func = self._create_agent_node(current, agent_config)
                    graph.add_node(current, router_func)
                    nodes_created.add(current)
                    graph.add_edge(prev_lg_node, current)
                    prev_lg_node = current
                    logger.info(f"Created router node for: {current}")

                # Get parallel targets
                parallel_targets = outgoing[current]
                parallel_target_configs = [
                    agent_registry.get(t, {})
                    for t in parallel_targets
                    if t in agent_registry
                ]

                # Find the merge target for this parallel group
                merge_target_for_group = None
                for target in parallel_targets:
                    for next_node in outgoing.get(target, []):
                        if next_node in merge_targets:
                            merge_target_for_group = next_node
                            break
                    if merge_target_for_group:
                        break

                # Create parallel Crew node for the targets
                if self._use_crewai and self._crewai_adapter and parallel_target_configs:
                    parallel_crew_counter += 1
                    parallel_node_name = f"parallel_crew_{parallel_crew_counter}"

                    parallel_func = self._crewai_adapter.create_parallel_crew_node(
                        agent_configs=parallel_target_configs,
                        aggregation_strategy="combine"
                    )
                    graph.add_node(parallel_node_name, parallel_func)
                    nodes_created.add(parallel_node_name)
                    graph.add_edge(prev_lg_node, parallel_node_name)

                    logger.info(f"Created parallel Crew '{parallel_node_name}' with {len(parallel_targets)} agents: {parallel_targets}")

                    # Mark parallel targets as visited (they're in the Crew)
                    for target in parallel_targets:
                        visited.add(target)

                    # If there's a merge target, record the connection
                    if merge_target_for_group:
                        merge_target_to_parallel_crew[merge_target_for_group] = parallel_node_name
                        # Queue the merge target for processing
                        queue.append((merge_target_for_group, parallel_node_name))
                    else:
                        # No merge target - parallel targets might be terminal
                        # Check if any parallel target has outgoing connections
                        has_continuation = False
                        for target in parallel_targets:
                            for next_node in outgoing.get(target, []):
                                if next_node not in visited:
                                    queue.append((next_node, parallel_node_name))
                                    has_continuation = True

                        if not has_continuation:
                            # Parallel targets are terminal
                            graph.add_edge(parallel_node_name, END)
                            logger.info(f"Parallel crew {parallel_node_name} connects to END")
                else:
                    # Fallback: create individual nodes for each target
                    logger.warning("CrewAI not available, creating individual nodes for parallel targets")
                    for target in parallel_targets:
                        if target not in visited:
                            queue.append((target, prev_lg_node))

            # =====================================================================
            # CASE B: Current node is a MERGE TARGET (receives from multiple sources)
            # =====================================================================
            elif current in merge_targets:
                logger.info(f"Processing merge target: {current}")

                # Create the merge target as a sequential node
                if current not in nodes_created:
                    merge_func = self._create_agent_node(current, agent_config)
                    graph.add_node(current, merge_func)
                    nodes_created.add(current)

                    # Connect from the parallel crew that feeds into this merge target
                    if current in merge_target_to_parallel_crew:
                        parallel_crew_name = merge_target_to_parallel_crew[current]
                        graph.add_edge(parallel_crew_name, current)
                        logger.info(f"Connected {parallel_crew_name} -> {current}")
                    else:
                        # Fallback: connect from prev_lg_node
                        graph.add_edge(prev_lg_node, current)

                # Queue outgoing nodes
                for next_node in outgoing.get(current, []):
                    if next_node not in visited:
                        queue.append((next_node, current))

                # If terminal, connect to END
                if current in terminal_nodes:
                    graph.add_edge(current, END)
                    logger.info(f"Terminal merge target {current} connects to END")

            # =====================================================================
            # CASE C: Regular sequential node
            # =====================================================================
            else:
                logger.info(f"Processing sequential node: {current}")

                if current not in nodes_created:
                    # Create agent node
                    node_func = self._create_agent_node(current, agent_config)
                    graph.add_node(current, node_func)
                    nodes_created.add(current)
                    # Only add edge if this node is NOT a conditional target
                    # (conditional targets already have edges via add_conditional_edges)
                    if current not in conditional_targets:
                        graph.add_edge(prev_lg_node, current)
                    logger.info(f"Created sequential node: {current}")

                # Queue outgoing nodes
                for next_node in outgoing.get(current, []):
                    if next_node not in visited:
                        queue.append((next_node, current))

                # If terminal, connect to END
                if current in terminal_nodes:
                    graph.add_edge(current, END)
                    logger.info(f"Terminal node {current} connects to END")

        # =====================================================================
        # STEP 6: Verify graph completeness
        # =====================================================================
        unvisited = set(agents) - visited
        if unvisited:
            logger.warning(f"Some agents were not visited during traversal: {unvisited}")
            # These might be disconnected nodes - add them with edge to END
            for agent_id in unvisited:
                if agent_id not in nodes_created:
                    agent_config = agent_registry.get(agent_id, {})
                    node_func = self._create_agent_node(agent_id, agent_config)
                    graph.add_node(agent_id, node_func)
                    nodes_created.add(agent_id)
                    graph.add_edge("coordinator", agent_id)
                    graph.add_edge(agent_id, END)
                    logger.info(f"Added disconnected node {agent_id} with direct path to END")

        logger.info(f"Hybrid workflow graph complete: {len(nodes_created)} nodes created")

        # =====================================================================
        # STEP 7: Compile with memory checkpointer
        # =====================================================================
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    def _create_conditional_node(
        self,
        node_id: str,
        agent_config: Dict[str, Any]
    ):
        """
        Create a conditional/router node function.

        This node evaluates conditions and stores the routing decision in state.
        The actual routing is handled by LangGraph's add_conditional_edges.

        Args:
            node_id: Node identifier
            agent_config: Node configuration with branches

        Returns:
            Callable node function that evaluates conditions
        """
        node_name = agent_config.get("name", node_id)
        branches_config = agent_config.get("config", {}).get("branches", [])

        def conditional_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Conditional node that evaluates branch conditions.

            The routing decision is stored in state for the routing function
            to use when LangGraph invokes add_conditional_edges.
            """
            logger.info(f"Conditional node '{node_name}' evaluating branches")

            # Evaluate each branch condition against current state
            selected_branch = None
            selected_target = None

            for branch in branches_config:
                branch_type = branch.get("type", "")
                condition = branch.get("condition", "")
                target_node_id = branch.get("targetNodeId")

                if branch_type == "else":
                    # Else branch is the fallback - save it but continue checking
                    if selected_branch is None:
                        selected_branch = branch
                        selected_target = str(target_node_id) if target_node_id else None
                    continue

                if branch_type == "if" and condition:
                    # Evaluate the condition against state
                    try:
                        # Create evaluation context from state
                        eval_context = dict(state)
                        # Safe evaluation of simple conditions
                        result = self._evaluate_condition(condition, eval_context)
                        if result:
                            selected_branch = branch
                            selected_target = str(target_node_id) if target_node_id else None
                            logger.info(f"Condition '{condition}' evaluated to True, routing to {selected_target}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to evaluate condition '{condition}': {e}")
                        continue

            # If no condition matched and we have an else branch, use it
            if selected_target is None and branches_config:
                # Find the else branch
                for branch in branches_config:
                    if branch.get("type") == "else":
                        selected_target = str(branch.get("targetNodeId"))
                        logger.info(f"No conditions matched, using else branch to {selected_target}")
                        break

            # Store routing decision in state for the routing function
            return {
                "_conditional_route": selected_target,
                "_conditional_node": node_id,
                "messages": [{
                    "node": node_id,
                    "type": "conditional",
                    "name": node_name,
                    "selected_route": selected_target,
                    "action": "branch_evaluated"
                }]
            }

        return conditional_node

    def _create_conditional_routing_function(
        self,
        node_id: str,
        agent_config: Dict[str, Any],
        branch_targets: List[str],
        branches_config: List[Dict[str, Any]]
    ):
        """
        Create routing function for LangGraph conditional edges.

        This function is called by LangGraph to determine which branch to take.
        It reads the routing decision from state (set by _create_conditional_node).

        Args:
            node_id: Conditional node identifier
            agent_config: Node configuration
            branch_targets: List of possible target node IDs
            branches_config: Branch configurations with conditions

        Returns:
            Callable routing function that returns the target node ID
        """
        # Determine default target (else branch or last target)
        default_target = None
        for branch in branches_config:
            if branch.get("type") == "else":
                target = branch.get("targetNodeId")
                default_target = str(target) if target else None
                break

        if default_target is None and branch_targets:
            default_target = branch_targets[-1]

        def routing_function(state: Dict[str, Any]) -> str:
            """
            Routing function that returns the target node ID.

            This reads the decision made by the conditional node and returns
            the appropriate target for LangGraph to route to.
            """
            # Get the routing decision from state
            selected_route = state.get("_conditional_route")

            if selected_route and selected_route in branch_targets:
                logger.info(f"Routing from {node_id} to {selected_route}")
                return selected_route

            # Fallback: evaluate conditions directly if not in state
            # This handles cases where the conditional node result isn't in state
            for branch in branches_config:
                branch_type = branch.get("type", "")
                condition = branch.get("condition", "")
                target_node_id = branch.get("targetNodeId")

                if branch_type == "if" and condition:
                    try:
                        eval_context = dict(state)
                        result = self._evaluate_condition(condition, eval_context)
                        if result:
                            target = str(target_node_id) if target_node_id else None
                            if target and target in branch_targets:
                                logger.info(f"Direct condition evaluation: routing to {target}")
                                return target
                    except Exception as e:
                        logger.warning(f"Routing function condition eval failed: {e}")
                        continue

            # Use default target
            logger.info(f"Using default route from {node_id} to {default_target}")
            return default_target

        return routing_function

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate a condition string against a context.

        Supports simple conditions like:
        - "win_probability < 30"
        - "status == 'approved'"
        - "win_probability < 30 AND opportunity_value < 500000"

        Args:
            condition: Condition string to evaluate
            context: Dictionary of variable values

        Returns:
            Boolean result of condition evaluation
        """
        if not condition or not condition.strip():
            return False

        # Normalize the condition
        normalized = condition.strip()

        # Handle AND/OR operators (case insensitive)
        normalized = normalized.replace(" AND ", " and ").replace(" OR ", " or ")

        # Create a safe evaluation environment with only the context variables
        # and basic comparison operators
        safe_globals = {"__builtins__": {}}
        safe_locals = dict(context)

        # Add common boolean values
        safe_locals["True"] = True
        safe_locals["False"] = False
        safe_locals["true"] = True
        safe_locals["false"] = False
        safe_locals["None"] = None
        safe_locals["null"] = None

        try:
            # Evaluate the condition
            result = eval(normalized, safe_globals, safe_locals)
            return bool(result)
        except NameError as e:
            # Variable not found in context - treat as False
            logger.debug(f"Variable not found in condition '{condition}': {e}")
            return False
        except Exception as e:
            logger.warning(f"Condition evaluation error for '{condition}': {e}")
            return False

    def _create_agent_node(
        self,
        agent_id: str,
        agent_config: Dict[str, Any]
    ):
        """
        Create agent node function with CrewAI or direct LLM execution and HITL support.

        ARCHITECTURE:
        - This creates a FUNCTION that LangGraph will call
        - Inside this function, we can use CrewAI for agent execution
        - CrewAI is invoked INSIDE the function, not for graph control

        Args:
            agent_id: Agent identifier
            agent_config: Agent configuration

        Returns:
            Callable node function compatible with LangGraph
        """
        # Check if this is a HITL node
        node_type = agent_config.get("metadata", {}).get("node_type")
        is_hitl_node = (node_type == "HITL")

        # If CrewAI is enabled, use CrewAI adapter for agent execution
        if self._use_crewai and self._crewai_adapter and not is_hitl_node:
            logger.info(f"Creating CrewAI-powered node for agent: {agent_id}")
            return self._crewai_adapter.create_sequential_agent_node(agent_config)

        # Otherwise, use legacy direct LLM execution
        logger.info(f"Creating direct LLM node for agent: {agent_id}")

        def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Agent node execution function with real LLM calls and HITL support.

            Args:
                state: Current workflow state

            Returns:
                Updated state
            """
            from ..runtime.hitl import HITLManager

            # Get agent configuration
            agent_name = agent_config.get("name", agent_id)
            agent_role = agent_config.get("role", "Processing")
            agent_description = agent_config.get("description", "")
            input_schema = agent_config.get("input_schema", [])
            output_schema = agent_config.get("output_schema", [])
            llm_config = agent_config.get("llm", {})

            # Extract inputs from state
            inputs = {key: state.get(key) for key in input_schema if key in state}

            # If this is a HITL node, request approval BEFORE executing
            if is_hitl_node:
                hitl = HITLManager()

                # Get workflow and run IDs from state
                workflow_id = state.get("workflow_id", "unknown")
                run_id = state.get("run_id", "unknown")

                # Request approval
                context = {
                    "agent_output": {"pending": "awaiting approval before execution"},
                    "state_snapshot": dict(state),
                    "inputs": inputs
                }

                interrupt_info = hitl.request_approval(
                    run_id=run_id,
                    workflow_id=workflow_id,
                    blocked_at=agent_id,
                    context=context
                )

                # Check HITL status - execution pauses here
                status = hitl.get_status(run_id)

                if status.get("state") == "rejected":
                    # Execution was rejected
                    raise RuntimeError(f"Workflow rejected at HITL checkpoint: {agent_id}")

                elif status.get("state") == "modified":
                    # Agent was modified - reload configuration
                    # In production, reload agent config here
                    pass

            # Build prompt for LLM (only if not a pure HITL node)
            if not is_hitl_node or llm_config:
                prompt = f"""You are {agent_name}, a specialized agent with the following role:
{agent_role}

{agent_description}

Your task is to process the following inputs and generate outputs:

Inputs:
{inputs}

Please provide your response in a clear, structured format. Focus on your specific role and responsibilities."""

                # Execute real LLM call
                try:
                    llm_response = self._execute_llm_call(llm_config, prompt)

                    # Create outputs based on LLM response
                    outputs = {}
                    for key in output_schema:
                        outputs[key] = llm_response

                except Exception as e:
                    # Fallback if LLM call fails
                    outputs = {
                        key: f"Error in {agent_name}: {str(e)}" for key in output_schema
                    }
            else:
                # HITL node - just pass through
                outputs = {key: state.get(key) for key in output_schema if key in state}

            # Return only new data - LangGraph merges via state schema
            # messages uses operator.add, so return ONLY the new message
            return {
                **outputs,
                "crew_result": next(iter(outputs.values()), "") if outputs else "",
                "messages": [{
                    "agent": agent_id,
                    "role": agent_role,
                    "inputs": inputs,
                    "outputs": outputs,
                    "hitl_checkpoint": is_hitl_node
                }]
            }

        return agent_node

    def _execute_llm_call(self, llm_config: Dict[str, Any], prompt: str) -> str:
        """
        Execute actual LLM call based on provider.

        Args:
            llm_config: LLM configuration (provider, model, temperature)
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        provider = llm_config.get("provider", os.getenv("LLM_PROVIDER", "openrouter"))
        model = llm_config.get(
            "model",
            os.getenv("OPENROUTER_MODEL", "allenai/molmo-2-8b:free")
        )
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 1000)

        if provider == "openrouter":
            return self._call_openrouter(model, prompt, temperature, max_tokens)
        elif provider == "openai":
            return self._call_openai(model, prompt, temperature, max_tokens)
        elif provider == "anthropic":
            return self._call_anthropic(model, prompt, temperature, max_tokens)
        elif provider == "azure":
            return self._call_azure(model, prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _call_openai(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API or Ollama using ChatOpenAI."""
        try:
            from langchain_openai import ChatOpenAI

            # Check if using Ollama
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.188.100.131:8004/v1")
            use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"

            if use_ollama:
                # Use Ollama endpoint
                llm = ChatOpenAI(
                    base_url=ollama_url,
                    api_key="ollama",
                    model=os.getenv("OLLAMA_MODEL", "mistral-nemo:12b-instruct-2407-fp16"),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            response = llm.invoke(prompt)
            return response.content

        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _call_anthropic(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API."""
        try:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")

    def _call_azure(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Azure OpenAI API."""
        try:
            from openai import AzureOpenAI

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

            if not api_key or not endpoint:
                raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")

            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API call failed: {e}")

    def _call_openrouter(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenRouter via ChatOpenAI wrapper."""
        try:
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")

            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed: {e}")

    def _determine_graph_type(self, execution_model: str) -> str:
        """
        Determine LangGraph graph type from execution model.

        Args:
            execution_model: Execution model

        Returns:
            LangGraph graph type
        """
        mapping = {
            "sequential": "StateGraph",
            "parallel": "StateGraph",
            "hierarchical": "StateGraph",
            "hybrid": "StateGraph"
        }
        return mapping.get(execution_model, "StateGraph")
