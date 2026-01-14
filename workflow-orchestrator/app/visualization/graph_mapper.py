"""
Graph Mapper
Maps workflow JSON to UI graph format compatible with React Flow / similar libraries
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.schemas.api_models import AgentSystemDesign
from app.workflow.graph_builder import GraphBuilder, WorkflowGraph, GraphNode, GraphEdge
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# UI GRAPH MODELS
# ============================================================================

@dataclass
class UINode:
    """
    UI-specific node format

    Compatible with React Flow and similar graph libraries
    """
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]
    style: Optional[Dict[str, Any]] = None
    className: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "data": self.data
        }
        if self.style:
            result["style"] = self.style
        if self.className:
            result["className"] = self.className
        return result


@dataclass
class UIEdge:
    """
    UI-specific edge format

    Compatible with React Flow and similar graph libraries
    """
    id: str
    source: str
    target: str
    type: Optional[str] = "default"
    label: Optional[str] = None
    animated: bool = False
    style: Optional[Dict[str, Any]] = None
    markerEnd: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "source": self.source,
            "target": self.target
        }
        if self.type:
            result["type"] = self.type
        if self.label:
            result["label"] = self.label
        if self.animated:
            result["animated"] = self.animated
        if self.style:
            result["style"] = self.style
        if self.markerEnd:
            result["markerEnd"] = self.markerEnd
        return result


@dataclass
class UIGraph:
    """
    Complete UI graph

    Ready for frontend rendering
    """
    nodes: List[UINode]
    edges: List[UIEdge]
    viewport: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "viewport": self.viewport,
            "metadata": self.metadata
        }


# ============================================================================
# GRAPH MAPPER
# ============================================================================

class GraphMapper:
    """
    Maps workflow JSON to UI graph format

    Converts internal graph representation to frontend-compatible format
    with proper positioning, styling, and metadata

    Usage:
        mapper = GraphMapper()
        ui_graph = mapper.map_to_ui_graph(agent_system)

        # Send to frontend
        return JSONResponse(ui_graph.to_dict())
    """

    # Node type styling
    NODE_STYLES = {
        "start": {
            "background": "#10b981",
            "color": "#ffffff",
            "border": "2px solid #059669",
            "borderRadius": "50%",
            "width": "80px",
            "height": "80px"
        },
        "end": {
            "background": "#ef4444",
            "color": "#ffffff",
            "border": "2px solid #dc2626",
            "borderRadius": "50%",
            "width": "80px",
            "height": "80px"
        },
        "agent": {
            "background": "#3b82f6",
            "color": "#ffffff",
            "border": "2px solid #2563eb",
            "borderRadius": "8px",
            "padding": "12px",
            "minWidth": "150px"
        },
        "master_agent": {
            "background": "#8b5cf6",
            "color": "#ffffff",
            "border": "3px solid #7c3aed",
            "borderRadius": "8px",
            "padding": "12px",
            "minWidth": "180px"
        }
    }

    # Edge type styling
    EDGE_STYLES = {
        "default": {
            "stroke": "#64748b",
            "strokeWidth": 2
        },
        "parallel": {
            "stroke": "#06b6d4",
            "strokeWidth": 2,
            "strokeDasharray": "5,5"
        },
        "hierarchical": {
            "stroke": "#8b5cf6",
            "strokeWidth": 2
        },
        "conditional": {
            "stroke": "#f59e0b",
            "strokeWidth": 2,
            "strokeDasharray": "3,3"
        },
        "data_flow": {
            "stroke": "#10b981",
            "strokeWidth": 2
        }
    }

    def __init__(self):
        """Initialize graph mapper"""
        self.graph_builder = GraphBuilder()
        logger.info("Graph mapper initialized")

    def map_to_ui_graph(
        self,
        agent_system: AgentSystemDesign,
        layout_algorithm: str = "dagre"
    ) -> UIGraph:
        """
        Map agent system to UI graph

        Args:
            agent_system: Agent system design
            layout_algorithm: Layout algorithm to use

        Returns:
            UIGraph ready for frontend
        """
        logger.info(f"Mapping workflow to UI graph: {agent_system.system_name}")

        # Build internal graph
        internal_graph = self.graph_builder.build_graph(agent_system)

        # Apply layout
        from app.visualization.layout import apply_layout
        positioned_graph = apply_layout(internal_graph, algorithm=layout_algorithm)

        # Convert to UI format
        ui_nodes = self._convert_nodes(positioned_graph.nodes)
        ui_edges = self._convert_edges(positioned_graph.edges)

        # Build viewport (default centered)
        viewport = {
            "x": 0,
            "y": 0,
            "zoom": 1.0
        }

        # Add metadata
        metadata = {
            **positioned_graph.metadata,
            "layout_algorithm": layout_algorithm,
            "node_count": len(ui_nodes),
            "edge_count": len(ui_edges)
        }

        ui_graph = UIGraph(
            nodes=ui_nodes,
            edges=ui_edges,
            viewport=viewport,
            metadata=metadata
        )

        logger.info(f"Mapped UI graph: {len(ui_nodes)} nodes, {len(ui_edges)} edges")

        return ui_graph

    def _convert_nodes(self, nodes: List[GraphNode]) -> List[UINode]:
        """
        Convert internal nodes to UI nodes

        Args:
            nodes: Internal graph nodes

        Returns:
            List of UI nodes
        """
        ui_nodes = []

        for node in nodes:
            # Get position (should be set by layout algorithm)
            position = node.position or {"x": 0, "y": 0}

            # Get style for node type
            style = self.NODE_STYLES.get(node.type, self.NODE_STYLES["agent"])

            # Build UI node data
            ui_data = {
                "label": node.label,
                **node.data
            }

            ui_node = UINode(
                id=node.id,
                type=node.type,
                position=position,
                data=ui_data,
                style=style,
                className=f"node-{node.type}"
            )

            ui_nodes.append(ui_node)

        return ui_nodes

    def _convert_edges(self, edges: List[GraphEdge]) -> List[UIEdge]:
        """
        Convert internal edges to UI edges

        Args:
            edges: Internal graph edges

        Returns:
            List of UI edges
        """
        ui_edges = []

        for edge in edges:
            # Get style for edge type
            style = self.EDGE_STYLES.get(edge.type, self.EDGE_STYLES["default"])

            # Determine if animated
            animated = edge.type in ["parallel", "data_flow"]

            # Arrow marker
            marker_end = {
                "type": "arrowclosed",
                "color": style.get("stroke", "#64748b")
            }

            ui_edge = UIEdge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                type=edge.type,
                label=edge.label,
                animated=animated,
                style=style,
                markerEnd=marker_end
            )

            ui_edges.append(ui_edge)

        return ui_edges


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def map_workflow_to_ui_graph(
    agent_system: AgentSystemDesign,
    layout: str = "dagre"
) -> Dict[str, Any]:
    """
    Convenience function to map workflow to UI graph

    Args:
        agent_system: Agent system design
        layout: Layout algorithm

    Returns:
        UI graph dictionary
    """
    mapper = GraphMapper()
    ui_graph = mapper.map_to_ui_graph(agent_system, layout)
    return ui_graph.to_dict()


def convert_graph_to_json(workflow_graph: WorkflowGraph) -> str:
    """
    Convert workflow graph to JSON string

    Args:
        workflow_graph: Workflow graph

    Returns:
        JSON string
    """
    import json
    return json.dumps(workflow_graph.to_dict(), indent=2)
