"""
Layout Algorithms
Auto-layout algorithms for positioning workflow graph nodes
"""
from typing import Dict, List, Tuple
import math

from app.workflow.graph_builder import WorkflowGraph, GraphNode, GraphEdge
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# LAYOUT CONSTANTS
# ============================================================================

# Node dimensions
NODE_WIDTH = 200
NODE_HEIGHT = 80
START_END_SIZE = 80

# Spacing
HORIZONTAL_SPACING = 150
VERTICAL_SPACING = 120
LAYER_SPACING = 250


# ============================================================================
# DAGRE LAYOUT (Layered Directed Acyclic Graph)
# ============================================================================

def dagre_layout(graph: WorkflowGraph) -> WorkflowGraph:
    """
    Apply Dagre layout algorithm (layered DAG layout)

    Best for: Sequential, hierarchical, and DAG workflows

    Args:
        graph: Workflow graph

    Returns:
        Graph with positioned nodes
    """
    logger.debug("Applying Dagre layout")

    # Build adjacency list
    adj_list = _build_adjacency_list(graph)

    # Perform topological sort to get layers
    layers = _topological_layers(graph, adj_list)

    # Position nodes in layers
    for layer_index, layer_nodes in enumerate(layers):
        layer_width = len(layer_nodes) * (NODE_WIDTH + HORIZONTAL_SPACING)
        start_x = -layer_width / 2

        for node_index, node_id in enumerate(layer_nodes):
            node = _get_node_by_id(graph, node_id)
            if node:
                # Calculate position
                x = start_x + node_index * (NODE_WIDTH + HORIZONTAL_SPACING) + NODE_WIDTH / 2
                y = layer_index * LAYER_SPACING

                node.position = {"x": x, "y": y}

    return graph


def _topological_layers(graph: WorkflowGraph, adj_list: Dict[str, List[str]]) -> List[List[str]]:
    """
    Compute topological layers (nodes at same depth)

    Args:
        graph: Workflow graph
        adj_list: Adjacency list

    Returns:
        List of layers, each layer is list of node IDs
    """
    # Calculate in-degree for each node
    in_degree = {node.id: 0 for node in graph.nodes}
    for node_id in adj_list:
        for neighbor in adj_list[node_id]:
            in_degree[neighbor] += 1

    # Find nodes with no incoming edges (layer 0)
    layers = []
    current_layer = [node_id for node_id, degree in in_degree.items() if degree == 0]
    layers.append(current_layer)

    # Process layers
    visited = set(current_layer)

    while current_layer:
        next_layer = []

        for node_id in current_layer:
            for neighbor in adj_list.get(node_id, []):
                if neighbor not in visited:
                    # Check if all parents are visited
                    parents_visited = all(
                        pred in visited
                        for pred in _get_predecessors(graph, neighbor)
                    )
                    if parents_visited:
                        next_layer.append(neighbor)
                        visited.add(neighbor)

        if next_layer:
            layers.append(next_layer)

        current_layer = next_layer

    return layers


# ============================================================================
# FORCE-DIRECTED LAYOUT
# ============================================================================

def force_layout(graph: WorkflowGraph, iterations: int = 100) -> WorkflowGraph:
    """
    Apply force-directed layout (spring/physics-based)

    Best for: General graphs, clusters, organic layouts

    Args:
        graph: Workflow graph
        iterations: Number of simulation iterations

    Returns:
        Graph with positioned nodes
    """
    logger.debug(f"Applying force layout ({iterations} iterations)")

    # Initialize random positions
    for i, node in enumerate(graph.nodes):
        angle = (i / len(graph.nodes)) * 2 * math.pi
        radius = 300
        node.position = {
            "x": radius * math.cos(angle),
            "y": radius * math.sin(angle)
        }

    # Run force simulation
    for iteration in range(iterations):
        _apply_forces(graph)

    return graph


def _apply_forces(graph: WorkflowGraph):
    """
    Apply repulsive and attractive forces to nodes

    Args:
        graph: Workflow graph
    """
    k = 150  # Optimal distance
    repulsion_strength = 5000
    attraction_strength = 0.1

    # Calculate repulsive forces between all nodes
    for i, node1 in enumerate(graph.nodes):
        force_x = 0
        force_y = 0

        for j, node2 in enumerate(graph.nodes):
            if i != j:
                dx = node1.position["x"] - node2.position["x"]
                dy = node1.position["y"] - node2.position["y"]
                distance = math.sqrt(dx * dx + dy * dy) or 1

                # Repulsive force (Coulomb's law)
                repulsion = repulsion_strength / (distance * distance)
                force_x += (dx / distance) * repulsion
                force_y += (dy / distance) * repulsion

        # Store forces
        node1.data["force_x"] = force_x
        node1.data["force_y"] = force_y

    # Calculate attractive forces along edges
    for edge in graph.edges:
        source_node = _get_node_by_id(graph, edge.source)
        target_node = _get_node_by_id(graph, edge.target)

        if source_node and target_node:
            dx = target_node.position["x"] - source_node.position["x"]
            dy = target_node.position["y"] - source_node.position["y"]
            distance = math.sqrt(dx * dx + dy * dy) or 1

            # Attractive force (Hooke's law)
            attraction = attraction_strength * (distance - k)

            force_x = (dx / distance) * attraction
            force_y = (dy / distance) * attraction

            source_node.data["force_x"] += force_x
            source_node.data["force_y"] += force_y
            target_node.data["force_x"] -= force_x
            target_node.data["force_y"] -= force_y

    # Update positions
    damping = 0.5
    for node in graph.nodes:
        node.position["x"] += node.data.get("force_x", 0) * damping
        node.position["y"] += node.data.get("force_y", 0) * damping


# ============================================================================
# HIERARCHICAL LAYOUT
# ============================================================================

def hierarchical_layout(graph: WorkflowGraph) -> WorkflowGraph:
    """
    Apply hierarchical layout (tree-based)

    Best for: Hierarchical workflows with master-worker pattern

    Args:
        graph: Workflow graph

    Returns:
        Graph with positioned nodes
    """
    logger.debug("Applying hierarchical layout")

    # Find root node (master agent or start)
    root = _find_root_node(graph)

    if root:
        # Build tree structure
        tree = _build_tree(graph, root.id)

        # Position nodes in tree
        _position_tree(tree, 0, 0, graph)

    return graph


def _build_tree(graph: WorkflowGraph, root_id: str) -> Dict:
    """
    Build tree structure from graph

    Args:
        graph: Workflow graph
        root_id: Root node ID

    Returns:
        Tree structure
    """
    visited = set()

    def build_subtree(node_id: str) -> Dict:
        visited.add(node_id)

        children = []
        for edge in graph.edges:
            if edge.source == node_id and edge.target not in visited:
                child = build_subtree(edge.target)
                children.append(child)

        return {
            "id": node_id,
            "children": children
        }

    return build_subtree(root_id)


def _position_tree(tree: Dict, x: float, y: float, graph: WorkflowGraph, level: int = 0):
    """
    Position nodes in tree

    Args:
        tree: Tree structure
        x: X coordinate
        y: Y coordinate
        graph: Workflow graph
        level: Tree level
    """
    node = _get_node_by_id(graph, tree["id"])
    if node:
        node.position = {"x": x, "y": y}

    # Position children
    children = tree.get("children", [])
    if children:
        total_width = len(children) * (NODE_WIDTH + HORIZONTAL_SPACING)
        start_x = x - total_width / 2

        for i, child in enumerate(children):
            child_x = start_x + i * (NODE_WIDTH + HORIZONTAL_SPACING) + NODE_WIDTH / 2
            child_y = y + LAYER_SPACING
            _position_tree(child, child_x, child_y, graph, level + 1)


# ============================================================================
# CIRCULAR LAYOUT
# ============================================================================

def circular_layout(graph: WorkflowGraph) -> WorkflowGraph:
    """
    Apply circular layout

    Best for: Small workflows, visual aesthetics

    Args:
        graph: Workflow graph

    Returns:
        Graph with positioned nodes
    """
    logger.debug("Applying circular layout")

    # Exclude start/end nodes
    agent_nodes = [n for n in graph.nodes if n.type not in ["start", "end"]]
    radius = max(300, len(agent_nodes) * 50)

    # Position agent nodes in circle
    for i, node in enumerate(agent_nodes):
        angle = (i / len(agent_nodes)) * 2 * math.pi
        node.position = {
            "x": radius * math.cos(angle),
            "y": radius * math.sin(angle)
        }

    # Position start/end nodes
    start_node = _get_node_by_id(graph, "start")
    end_node = _get_node_by_id(graph, "end")

    if start_node:
        start_node.position = {"x": 0, "y": -radius - 150}

    if end_node:
        end_node.position = {"x": 0, "y": radius + 150}

    return graph


# ============================================================================
# LAYOUT SELECTOR
# ============================================================================

def apply_layout(graph: WorkflowGraph, algorithm: str = "dagre") -> WorkflowGraph:
    """
    Apply layout algorithm to graph

    Args:
        graph: Workflow graph
        algorithm: Layout algorithm ("dagre", "force", "hierarchical", "circular")

    Returns:
        Graph with positioned nodes

    Raises:
        ValueError: If algorithm is unknown
    """
    logger.info(f"Applying layout algorithm: {algorithm}")

    if algorithm == "dagre":
        return dagre_layout(graph)
    elif algorithm == "force":
        return force_layout(graph)
    elif algorithm == "hierarchical":
        return hierarchical_layout(graph)
    elif algorithm == "circular":
        return circular_layout(graph)
    else:
        logger.warning(f"Unknown layout algorithm: {algorithm}, using dagre")
        return dagre_layout(graph)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _build_adjacency_list(graph: WorkflowGraph) -> Dict[str, List[str]]:
    """
    Build adjacency list from graph edges

    Args:
        graph: Workflow graph

    Returns:
        Adjacency list
    """
    adj_list = {node.id: [] for node in graph.nodes}

    for edge in graph.edges:
        if edge.source in adj_list:
            adj_list[edge.source].append(edge.target)

    return adj_list


def _get_node_by_id(graph: WorkflowGraph, node_id: str) -> GraphNode:
    """Get node by ID"""
    for node in graph.nodes:
        if node.id == node_id:
            return node
    return None


def _get_predecessors(graph: WorkflowGraph, node_id: str) -> List[str]:
    """Get all predecessor node IDs"""
    return [edge.source for edge in graph.edges if edge.target == node_id]


def _find_root_node(graph: WorkflowGraph) -> GraphNode:
    """Find root node (node with no incoming edges)"""
    # Count incoming edges
    incoming = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        incoming[edge.target] += 1

    # Find node with no incoming edges (excluding start)
    for node in graph.nodes:
        if incoming[node.id] == 0 and node.id != "start":
            return node

    # If not found, return start node
    return _get_node_by_id(graph, "start")


def calculate_layout(graph: WorkflowGraph, algorithm: str = "dagre") -> Dict[str, Dict[str, float]]:
    """
    Calculate layout and return positions only

    Args:
        graph: Workflow graph
        algorithm: Layout algorithm

    Returns:
        Dictionary mapping node_id to position {x, y}
    """
    positioned_graph = apply_layout(graph, algorithm)

    positions = {}
    for node in positioned_graph.nodes:
        if node.position:
            positions[node.id] = node.position

    return positions
