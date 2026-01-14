"""
Internal Visualization API
Component-to-component graph operations
"""
from fastapi import APIRouter, HTTPException
from app.schemas.api_models import (
    WorkflowGraphRequest,
    WorkflowGraphResponse,
    ApplyGraphEditRequest,
    ApplyGraphEditResponse
)
from app.services.visualization_service import get_visualization_service

router = APIRouter(prefix="/api/internal/visualize", tags=["Internal-Visualize"])

# Get service instance
visualization_service = get_visualization_service()


@router.post("/graph", response_model=WorkflowGraphResponse)
async def generate_graph(request: WorkflowGraphRequest) -> WorkflowGraphResponse:
    """
    Internal API: Generate workflow graph

    Used by UI or other components needing graph representation
    """
    try:
        result = await visualization_service.generate_graph(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-edits", response_model=ApplyGraphEditResponse)
async def apply_graph_edits(request: ApplyGraphEditRequest) -> ApplyGraphEditResponse:
    """
    Internal API: Apply UI graph edits back to workflow JSON

    Used by workflow component after UI edits
    """
    try:
        result = await visualization_service.apply_graph_edits(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
