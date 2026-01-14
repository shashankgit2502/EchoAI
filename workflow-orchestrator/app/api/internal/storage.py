"""
Internal Storage API
Component-to-component persistence operations
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from app.schemas.api_models import (
    SaveWorkflowRequest,
    SaveWorkflowResponse,
    LoadWorkflowRequest,
    ListWorkflowsResponse,
    DeleteWorkflowRequest,
    ArchiveWorkflowRequest,
    CloneWorkflowRequest,
    AgentSystemDesign
)
from app.services.storage_service import get_storage_service

router = APIRouter(prefix="/api/internal/storage", tags=["Internal-Storage"])

# Get service instance
storage_service = get_storage_service()


@router.post("/save/draft", response_model=SaveWorkflowResponse)
def save_draft(request: SaveWorkflowRequest) -> SaveWorkflowResponse:
    """
    Internal API: Save workflow as DRAFT
    """
    try:
        result = storage_service.save_draft(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save/temp", response_model=SaveWorkflowResponse)
def save_temp(request: SaveWorkflowRequest) -> SaveWorkflowResponse:
    """
    Internal API: Save workflow as TEMP
    """
    try:
        result = storage_service.save_temp(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save/final", response_model=SaveWorkflowResponse)
def save_final(request: SaveWorkflowRequest) -> SaveWorkflowResponse:
    """
    Internal API: Save workflow as FINAL
    """
    try:
        result = storage_service.save_final(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load", response_model=Optional[AgentSystemDesign])
def load_workflow(request: LoadWorkflowRequest) -> Optional[AgentSystemDesign]:
    """
    Internal API: Load workflow from storage
    """
    try:
        result = storage_service.load_workflow(request)
        if not result:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clone", response_model=AgentSystemDesign)
def clone_workflow(request: CloneWorkflowRequest) -> AgentSystemDesign:
    """
    Internal API: Clone FINAL → DRAFT
    """
    try:
        result = storage_service.clone_final_to_draft(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
def delete_workflow(request: DeleteWorkflowRequest) -> dict:
    """
    Internal API: Delete workflow (draft/temp only)
    """
    try:
        result = storage_service.delete_workflow(request)
        if not result:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions/{workflow_id}", response_model=List[str])
def list_versions(workflow_id: str) -> List[str]:
    """
    Internal API: List workflow versions
    """
    try:
        result = storage_service.list_versions(workflow_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
