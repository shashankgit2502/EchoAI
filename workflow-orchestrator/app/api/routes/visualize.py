"""
External Visualization API Routes
User-facing workflow visualization endpoints
"""
from fastapi import APIRouter, HTTPException
from app.schemas.api_models import (
    WorkflowGraphRequest,
    WorkflowGraphResponse,
    ApplyGraphEditRequest,
    ApplyGraphEditResponse
)
from app.services.visualization_service import get_visualization_service

router = APIRouter(prefix="/visualize", tags=["Visualization"])

# Get service instance
visualization_service = get_visualization_service()


@router.post("/graph", response_model=WorkflowGraphResponse)
async def generate_workflow_graph(request: WorkflowGraphRequest) -> WorkflowGraphResponse:
    """
    Generate graph representation of workflow

    Returns nodes and edges for UI visualization
    """
    try:
        result = await visualization_service.generate_graph(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph generation failed: {str(e)}"
        )


@router.post("/apply-edits", response_model=ApplyGraphEditResponse)
async def apply_graph_edits(request: ApplyGraphEditRequest) -> ApplyGraphEditResponse:
    """
    Apply UI graph edits back to workflow JSON

    Allows visual workflow editing
    """
    try:
        result = await visualization_service.apply_graph_edits(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph edit failed: {str(e)}"
        )
