"""
Visualization Service
Service layer wrapper for workflow graph operations
Provides clean interface for internal API communication
"""
from typing import Optional
from app.schemas.api_models import (
    AgentSystemDesign,
    WorkflowGraphRequest,
    WorkflowGraphResponse,
    ApplyGraphEditRequest,
    ApplyGraphEditResponse
)
from app.visualization.graph_mapper import GraphMapper
from app.visualization.graph_editor import GraphEditor
from app.core.logging import get_logger

logger = get_logger(__name__)


class VisualizationService:
    """
    Service layer for visualization operations

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Async + idempotent
    - Ready for microservice extraction

    Responsibilities:
    - Workflow → Graph conversion
    - Graph editing (UI → Workflow JSON)
    - Layout generation
    """

    def __init__(self):
        """Initialize visualization service"""
        self._graph_mapper = GraphMapper()
        self._graph_editor = GraphEditor()
        logger.info("VisualizationService initialized")

    async def generate_graph(
        self,
        request: WorkflowGraphRequest
    ) -> WorkflowGraphResponse:
        """
        Generate graph representation of workflow

        Args:
            request: Graph generation request

        Returns:
            WorkflowGraphResponse with nodes and edges
        """
        logger.info(f"Generating graph for: {request.agent_system.system_name}")

        workflow_name = request.workflow_name
        if workflow_name is None and request.agent_system.workflows:
            workflow_name = request.agent_system.workflows[0].name

        # Convert workflow to graph
        graph = await self._graph_mapper.workflow_to_graph(
            request.agent_system,
            workflow_name
        )

        logger.info(f"Graph generated: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

        return WorkflowGraphResponse(
            nodes=graph["nodes"],
            edges=graph["edges"],
            layout=graph.get("layout", {})
        )

    async def apply_graph_edits(
        self,
        request: ApplyGraphEditRequest
    ) -> ApplyGraphEditResponse:
        """
        Apply UI graph edits back to workflow JSON

        Args:
            request: Graph edit request

        Returns:
            ApplyGraphEditResponse with updated workflow
        """
        logger.info(f"Applying graph edits to: {request.agent_system.system_name}")

        try:
            updated_system, changes = await self._graph_editor.apply_edits(
                request.agent_system,
                request.graph_edits
            )

            logger.info(f"Graph edits applied: {len(changes)} changes")

            return ApplyGraphEditResponse(
                updated_agent_system=updated_system,
                changes_applied=changes,
                validation_required=True  # Always require re-validation after edits
            )

        except Exception as e:
            logger.error(f"Graph edit failed: {e}", exc_info=True)
            raise


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_visualization_service: Optional[VisualizationService] = None


def get_visualization_service() -> VisualizationService:
    """
    Get singleton visualization service instance

    Returns:
        VisualizationService instance
    """
    global _visualization_service

    if _visualization_service is None:
        _visualization_service = VisualizationService()

    return _visualization_service
