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
        Infer hybrid topology from connections when not explicitly specified.

        FIXED: Properly detects parallel patterns by analyzing connection graph:
        - Parallel: One source node connects to multiple targets
        - Merge: Multiple source nodes connect to one target
        - Sequential: Linear chain of connections

        IMPORTANT: This function now also handles Start/End pseudo-nodes from canvas
        and properly filters them from agent execution.
        """
        from langgraph.graph import END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Inferring hybrid topology from connections...")

        agents = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        if not connections and not agents:
            raise ValueError("No agents or connections defined for hybrid workflow")

        # FIX: Normalize connections - handle Start/End pseudo-nodes from canvas
        # Canvas may include connections like {"from": "start_xxx", "to": "agt_xxx"}
        # We need to identify these and handle them properly
        normalized_connections = []
        start_node_id = None
        end_node_id = None

        for conn in connections:
            from_node = conn.get("from", "")
            to_node = conn.get("to", "")

            # Detect Start pseudo-node (not in agents list, connects TO agents)
            if from_node and from_node not in agents and to_node in agents:
                if from_node.lower().startswith("start") or "start" in from_node.lower():
                    start_node_id = from_node
                    logger.info(f"Detected Start pseudo-node: {from_node}")
                    # Don't add this connection, but remember start -> to_node
                    continue

            # Detect End pseudo-node (not in agents list, receives FROM agents)
            if to_node and to_node not in agents and from_node in agents:
                if to_node.lower().startswith("end") or "end" in to_node.lower():
                    end_node_id = to_node
                    logger.info(f"Detected End pseudo-node: {to_node}")
                    # Don't add this connection
                    continue

            # Only include connections between actual agents
            if from_node in agents and to_node in agents:
                normalized_connections.append(conn)

        # Use normalized connections for analysis (agent-to-agent only)
        if normalized_connections:
            connections = normalized_connections
            logger.info(f"Using {len(connections)} normalized agent-to-agent connections")

        # Analyze connection graph to detect parallel patterns
        from_counts = {}  # How many targets each source connects to
        to_counts = {}    # How many sources connect to each target

        for conn in connections:
            from_node = conn.get("from")
            to_node = conn.get("to")
            from_counts[from_node] = from_counts.get(from_node, 0) + 1
            to_counts[to_node] = to_counts.get(to_node, 0) + 1

        # Identify parallel sections (one source -> multiple targets)
        # FIX: Only consider nodes that are actual agents
        parallel_sources = {node for node, count in from_counts.items()
                           if count > 1 and node in agents}
        # Identify merge points (multiple sources -> one target)
        merge_targets = {node for node, count in to_counts.items()
                        if count > 1 and node in agents}

        logger.info(f"Detected parallel sources: {parallel_sources}, merge targets: {merge_targets}")
        logger.info(f"Connection analysis: from_counts={from_counts}, to_counts={to_counts}")

        # If we detect parallel patterns, use CrewAI parallel execution
        if parallel_sources and self._use_crewai and self._crewai_adapter:
            logger.info("Parallel pattern detected, using CrewAI parallel Crew")

            # Build parallel groups from connections
            parallel_groups = []
            for source in parallel_sources:
                parallel_targets = [conn.get("to") for conn in connections if conn.get("from") == source]
                parallel_agent_configs = [agent_registry.get(aid, {}) for aid in parallel_targets if aid in agent_registry]
                if parallel_agent_configs:
                    parallel_groups.append({
                        "source": source,
                        "agents": parallel_targets,
                        "configs": parallel_agent_configs
                    })

            # Find entry point (node with no incoming edges)
            all_to = {conn.get("to") for conn in connections}
            all_from = {conn.get("from") for conn in connections}
            entry_candidates = all_from - all_to
            entry_point = list(entry_candidates)[0] if entry_candidates else agents[0] if agents else None

            # Find terminal nodes (nodes with no outgoing edges)
            terminal_nodes = all_to - all_from

            # Create coordinator to preserve user input
            def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
                """Coordinator preserves user input for inferred hybrid execution."""
                original_input = state.get("user_input") or state.get("message") or state.get("task_description") or ""
                logger.info(f"Inferred hybrid coordinator with {len(parallel_groups)} parallel groups")
                return {
                    "original_user_input": original_input,
                    "messages": [{
                        "node": "coordinator",
                        "action": "inferred_hybrid_execution"
                    }]
                }

            graph.add_node("coordinator", coordinator)
            graph.set_entry_point("coordinator")

            # If entry point is a parallel source, create parallel Crew node
            if entry_point in parallel_sources and parallel_groups:
                group = parallel_groups[0]
                parallel_node_func = self._crewai_adapter.create_parallel_crew_node(
                    agent_configs=group["configs"],
                    aggregation_strategy="combine"
                )
                graph.add_node("parallel_execution", parallel_node_func)
                graph.add_edge("coordinator", "parallel_execution")

                # Connect to merge target or END
                if merge_targets:
                    merge_target = list(merge_targets)[0]
                    # Add merge target as sequential node after parallel
                    if merge_target in agent_registry:
                        merge_agent = agent_registry.get(merge_target, {})
                        merge_node_func = self._create_agent_node(merge_target, merge_agent)
                        graph.add_node(merge_target, merge_node_func)
                        graph.add_edge("parallel_execution", merge_target)

                        # Continue sequential chain from merge target
                        prev_node = merge_target
                        visited = {entry_point, merge_target} | set(group["agents"])

                        # Follow connections from merge target
                        for conn in connections:
                            if conn.get("from") == prev_node:
                                next_node = conn.get("to")
                                if next_node not in visited and next_node in agent_registry:
                                    agent = agent_registry.get(next_node, {})
                                    node_func = self._create_agent_node(next_node, agent)
                                    graph.add_node(next_node, node_func)
                                    graph.add_edge(prev_node, next_node)
                                    visited.add(next_node)
                                    prev_node = next_node

                        graph.add_edge(prev_node, END)
                    else:
                        graph.add_edge("parallel_execution", END)
                else:
                    graph.add_edge("parallel_execution", END)
            else:
                # Entry point is not parallel source - handle as sequential start
                if entry_point and entry_point in agent_registry:
                    agent = agent_registry.get(entry_point, {})
                    node_func = self._create_agent_node(entry_point, agent)
                    graph.add_node(entry_point, node_func)
                    graph.add_edge("coordinator", entry_point)
                    # Then continue building the graph
                    # (simplified - full implementation would trace all paths)
                    for node in terminal_nodes:
                        if node in agent_registry and node not in [entry_point]:
                            agent = agent_registry.get(node, {})
                            node_func = self._create_agent_node(node, agent)
                            graph.add_node(node, node_func)
                        graph.add_edge(node if node in agent_registry else entry_point, END)
                else:
                    graph.add_edge("coordinator", END)

        else:
            # No parallel patterns or CrewAI not available - use sequential
            logger.info("No parallel patterns detected or CrewAI unavailable, using sequential execution")

            for agent_id in agents:
                agent = agent_registry.get(agent_id, {})
                node_func = self._create_agent_node(agent_id, agent)
                graph.add_node(agent_id, node_func)

            if connections:
                graph.set_entry_point(connections[0].get("from"))
                for conn in connections:
                    graph.add_edge(conn.get("from"), conn.get("to"))
                # Find terminal nodes
                all_to = {conn.get("to") for conn in connections}
                all_from = {conn.get("from") for conn in connections}
                terminal_nodes = all_to - all_from
                for node in terminal_nodes:
                    graph.add_edge(node, END)
            elif agents:
                # Linear chain
                graph.set_entry_point(agents[0])
                for i in range(len(agents) - 1):
                    graph.add_edge(agents[i], agents[i + 1])
                graph.add_edge(agents[-1], END)

        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

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
