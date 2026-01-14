"""
Graph Builder
Converts workflow JSON into graph structure (nodes and edges)
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from app.schemas.api_models import AgentSystemDesign, WorkflowStep
from app.core.constants import CommunicationPattern
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# GRAPH STRUCTURES
# ============================================================================

@dataclass
class GraphNode:
    """
    Graph node representing an agent or workflow step

    Attributes:
        id: Unique node ID
        type: Node type (agent, start, end, decision, etc.)
        label: Display label
        data: Additional node data
        position: Optional position (x, y)
    """
    id: str
    type: str
    label: str
    data: Dict[str, Any]
    position: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "data": self.data
        }
        if self.position:
            result["position"] = self.position
        return result


@dataclass
class GraphEdge:
    """
    Graph edge representing workflow flow

    Attributes:
        id: Unique edge ID
        source: Source node ID
        target: Target node ID
        type: Edge type (default, conditional, parallel)
        label: Optional edge label
        data: Additional edge data
    """
    id: str
    source: str
    target: str
    type: str = "default"
    label: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type
        }
        if self.label:
            result["label"] = self.label
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class WorkflowGraph:
    """
    Complete workflow graph

    Attributes:
        nodes: List of graph nodes
        edges: List of graph edges
        metadata: Graph metadata
    """
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }


# ============================================================================
# GRAPH BUILDER
# ============================================================================

class GraphBuilder:
    """
    Converts workflow JSON into graph structure

    Handles different communication patterns:
    - Sequential: Linear chain of nodes
    - Parallel: Multiple parallel branches
    - Hierarchical: Master node with workers
    - Conditional: Decision nodes with branches
    - Graph: Arbitrary DAG structure

    Usage:
        builder = GraphBuilder()
        graph = builder.build_graph(agent_system)

        # Access nodes and edges
        for node in graph.nodes:
            print(f"Node: {node.id} ({node.type})")

        for edge in graph.edges:
            print(f"Edge: {edge.source} -> {edge.target}")
    """

    def __init__(self):
        """Initialize graph builder"""
        logger.info("Graph builder initialized")

    def build_graph(self, agent_system: AgentSystemDesign) -> WorkflowGraph:
        """
        Build graph from agent system

        Args:
            agent_system: Agent system design

        Returns:
            WorkflowGraph with nodes and edges
        """
        logger.info(f"Building graph for workflow: {agent_system.system_name}")

        # Create nodes for all agents
        nodes = self._create_agent_nodes(agent_system)

        # Add start and end nodes
        start_node = GraphNode(
            id="start",
            type="start",
            label="Start",
            data={"description": "Workflow start"}
        )
        end_node = GraphNode(
            id="end",
            type="end",
            label="End",
            data={"description": "Workflow end"}
        )

        nodes.insert(0, start_node)
        nodes.append(end_node)

        # Create edges based on communication pattern
        edges = self._create_edges(agent_system, nodes)

        # Build metadata
        metadata = {
            "workflow_name": agent_system.system_name,
            "communication_pattern": agent_system.communication_pattern,
            "agent_count": len(agent_system.agents),
            "step_count": len(agent_system.workflows[0].steps) if agent_system.workflows else 0
        }

        graph = WorkflowGraph(nodes=nodes, edges=edges, metadata=metadata)

        logger.info(
            f"Built graph: {len(nodes)} nodes, {len(edges)} edges "
            f"(pattern: {agent_system.communication_pattern})"
        )

        return graph

    def _create_agent_nodes(self, agent_system: AgentSystemDesign) -> List[GraphNode]:
        """
        Create nodes for all agents

        Args:
            agent_system: Agent system

        Returns:
            List of graph nodes
        """
        nodes = []

        for agent in agent_system.agents:
            node = GraphNode(
                id=agent.id,
                type="agent" if not agent.is_master else "master_agent",
                label=agent.role or agent.id,
                data={
                    "agent_id": agent.id,
                    "role": agent.role,
                    "is_master": agent.is_master,
                    "llm_model": agent.llm_config.model,
                    "tools": agent.tools,
                    "system_prompt": agent.system_prompt[:100] + "..." if len(agent.system_prompt) > 100 else agent.system_prompt
                }
            )
            nodes.append(node)

        return nodes

    def _create_edges(
        self,
        agent_system: AgentSystemDesign,
        nodes: List[GraphNode]
    ) -> List[GraphEdge]:
        """
        Create edges based on communication pattern

        Args:
            agent_system: Agent system
            nodes: Graph nodes

        Returns:
            List of graph edges
        """
        pattern = agent_system.communication_pattern

        if pattern == CommunicationPattern.SEQUENTIAL:
            return self._create_sequential_edges(agent_system, nodes)
        elif pattern == CommunicationPattern.PARALLEL:
            return self._create_parallel_edges(agent_system, nodes)
        elif pattern == CommunicationPattern.HIERARCHICAL:
            return self._create_hierarchical_edges(agent_system, nodes)
        elif pattern == CommunicationPattern.CONDITIONAL:
            return self._create_conditional_edges(agent_system, nodes)
        elif pattern == CommunicationPattern.GRAPH:
            return self._create_graph_edges(agent_system, nodes)
        else:
            logger.warning(f"Unknown pattern: {pattern}, defaulting to sequential")
            return self._create_sequential_edges(agent_system, nodes)

    def _create_sequential_edges(
        self,
        agent_system: AgentSystemDesign,
        nodes: List[GraphNode]
    ) -> List[GraphEdge]:
        """Create edges for sequential pattern (linear chain)"""
        edges = []

        if not agent_system.workflows:
            return edges

        workflow = agent_system.workflows[0]
        steps = workflow.steps

        # Connect start to first step
        if steps:
            edges.append(GraphEdge(
                id="start_to_first",
                source="start",
                target=steps[0].agent_id,
                type="default"
            ))

        # Connect steps sequentially
        for i in range(len(steps) - 1):
            edge = GraphEdge(
                id=f"step_{i}_to_{i+1}",
                source=steps[i].agent_id,
                target=steps[i + 1].agent_id,
                type="default",
                data={"step_index": i}
            )
            edges.append(edge)

        # Connect last step to end
        if steps:
            edges.append(GraphEdge(
                id="last_to_end",
                source=steps[-1].agent_id,
                target="end",
                type="default"
            ))

        return edges

    def _create_parallel_edges(
        self,
        agent_system: AgentSystemDesign,
        nodes: List[GraphNode]
    ) -> List[GraphEdge]:
        """Create edges for parallel pattern (fan-out, fan-in)"""
        edges = []

        if not agent_system.workflows:
            return edges

        workflow = agent_system.workflows[0]
        steps = workflow.steps

        # Group steps by parallel_with
        parallel_groups = self._group_parallel_steps(steps)

        # Connect start to all parallel groups
        for group_id, group_steps in parallel_groups.items():
            for step in group_steps:
                edges.append(GraphEdge(
                    id=f"start_to_{step.agent_id}",
                    source="start",
                    target=step.agent_id,
                    type="parallel",
                    label=f"Parallel group {group_id}"
                ))

        # Connect all steps to end
        for step in steps:
            edges.append(GraphEdge(
                id=f"{step.agent_id}_to_end",
                source=step.agent_id,
                target="end",
                type="parallel"
            ))

        return edges

    def _create_hierarchical_edges(
        self,
        agent_system: AgentSystemDesign,
        nodes: List[GraphNode]
    ) -> List[GraphEdge]:
        """Create edges for hierarchical pattern (master-worker)"""
        edges = []

        # Find master agent
        master_agent = next((a for a in agent_system.agents if a.is_master), None)
        if not master_agent:
            logger.warning("No master agent found for hierarchical pattern")
            return edges

        worker_agents = [a for a in agent_system.agents if not a.is_master]

        # Connect start to master
        edges.append(GraphEdge(
            id="start_to_master",
            source="start",
            target=master_agent.id,
            type="default"
        ))

        # Connect master to all workers
        for worker in worker_agents:
            edges.append(GraphEdge(
                id=f"master_to_{worker.id}",
                source=master_agent.id,
                target=worker.id,
                type="hierarchical",
                label="Delegate"
            ))

            # Connect worker back to master
            edges.append(GraphEdge(
                id=f"{worker.id}_to_master",
                source=worker.id,
                target=master_agent.id,
                type="hierarchical",
                label="Report"
            ))

        # Connect master to end
        edges.append(GraphEdge(
            id="master_to_end",
            source=master_agent.id,
            target="end",
            type="default"
        ))

        return edges

    def _create_conditional_edges(
        self,
        agent_system: AgentSystemDesign,
        nodes: List[GraphNode]
    ) -> List[GraphEdge]:
        """Create edges for conditional pattern (decision branches)"""
        edges = []

        if not agent_system.workflows:
            return edges

        workflow = agent_system.workflows[0]
        steps = workflow.steps

        # Connect start to first step
        if steps:
            edges.append(GraphEdge(
                id="start_to_first",
                source="start",
                target=steps[0].agent_id,
                type="default"
            ))

        # Create conditional branches
        for i, step in enumerate(steps):
            if step.condition:
                # This step has a condition - create branching
                edges.append(GraphEdge(
                    id=f"conditional_{i}",
                    source=step.agent_id,
                    target=steps[i + 1].agent_id if i + 1 < len(steps) else "end",
                    type="conditional",
                    label=step.condition[:50] + "..." if len(step.condition) > 50 else step.condition
                ))
            else:
                # Normal flow
                if i + 1 < len(steps):
                    edges.append(GraphEdge(
                        id=f"step_{i}_to_{i+1}",
                        source=step.agent_id,
                        target=steps[i + 1].agent_id,
                        type="default"
                    ))

        # Connect last step to end
        if steps:
            edges.append(GraphEdge(
                id="last_to_end",
                source=steps[-1].agent_id,
                target="end",
                type="default"
            ))

        return edges

    def _create_graph_edges(
        self,
        agent_system: AgentSystemDesign,
        nodes: List[GraphNode]
    ) -> List[GraphEdge]:
        """Create edges for arbitrary graph pattern (DAG)"""
        edges = []

        if not agent_system.workflows:
            return edges

        workflow = agent_system.workflows[0]
        steps = workflow.steps

        # Build dependency map from steps
        step_map = {step.agent_id: step for step in steps}

        # Find entry points (steps with no dependencies)
        entry_steps = [s for s in steps if not s.inputs]

        # Connect start to entry points
        for step in entry_steps:
            edges.append(GraphEdge(
                id=f"start_to_{step.agent_id}",
                source="start",
                target=step.agent_id,
                type="default"
            ))

        # Create edges based on input/output dependencies
        for step in steps:
            if step.inputs:
                # Parse inputs to find dependencies
                # Assuming inputs like: {"data": "agent_a.output"}
                for key, value in step.inputs.items():
                    if isinstance(value, str) and "." in value:
                        source_agent = value.split(".")[0]
                        if source_agent in step_map:
                            edges.append(GraphEdge(
                                id=f"{source_agent}_to_{step.agent_id}",
                                source=source_agent,
                                target=step.agent_id,
                                type="data_flow",
                                label=key
                            ))

        # Find exit points (steps with no outputs referenced)
        all_sources = {e.source for e in edges if e.source != "start"}
        all_targets = {e.target for e in edges if e.target != "end"}
        exit_agents = all_sources - all_targets

        # Connect exit points to end
        for agent_id in exit_agents:
            edges.append(GraphEdge(
                id=f"{agent_id}_to_end",
                source=agent_id,
                target="end",
                type="default"
            ))

        return edges

    def _group_parallel_steps(self, steps: List[WorkflowStep]) -> Dict[str, List[WorkflowStep]]:
        """
        Group steps by parallel_with attribute

        Args:
            steps: Workflow steps

        Returns:
            Dictionary mapping group ID to list of steps
        """
        groups: Dict[str, List[WorkflowStep]] = {}

        for step in steps:
            group_id = step.parallel_with or step.agent_id
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(step)

        return groups


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_workflow_graph(agent_system: AgentSystemDesign) -> WorkflowGraph:
    """
    Convenience function to build workflow graph

    Args:
        agent_system: Agent system design

    Returns:
        WorkflowGraph
    """
    builder = GraphBuilder()
    return builder.build_graph(agent_system)


def get_node_by_id(graph: WorkflowGraph, node_id: str) -> Optional[GraphNode]:
    """
    Get node by ID

    Args:
        graph: Workflow graph
        node_id: Node ID

    Returns:
        GraphNode or None
    """
    for node in graph.nodes:
        if node.id == node_id:
            return node
    return None


def get_outgoing_edges(graph: WorkflowGraph, node_id: str) -> List[GraphEdge]:
    """
    Get all outgoing edges from node

    Args:
        graph: Workflow graph
        node_id: Node ID

    Returns:
        List of outgoing edges
    """
    return [edge for edge in graph.edges if edge.source == node_id]


def get_incoming_edges(graph: WorkflowGraph, node_id: str) -> List[GraphEdge]:
    """
    Get all incoming edges to node

    Args:
        graph: Workflow graph
        node_id: Node ID

    Returns:
        List of incoming edges
    """
    return [edge for edge in graph.edges if edge.target == node_id]
