"""
Internal Workflow API
Component-to-component workflow operations
"""
from fastapi import APIRouter, HTTPException
from app.schemas.api_models import (
    CompileWorkflowRequest,
    CompileWorkflowResponse,
    ModifyWorkflowRequest,
    AgentSystemDesign,
    VersionWorkflowRequest,
    VersionWorkflowResponse
)
from app.services.workflow_service import get_workflow_service

router = APIRouter(prefix="/api/internal/workflow", tags=["Internal-Workflow"])

# Get service instance
workflow_service = get_workflow_service()


@router.post("/compile", response_model=CompileWorkflowResponse)
async def compile_workflow(request: CompileWorkflowRequest) -> CompileWorkflowResponse:
    """
    Internal API: Compile workflow JSON to LangGraph StateGraph

    Used by runtime component before execution
    """
    try:
        result = await workflow_service.compile_workflow(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/modify", response_model=AgentSystemDesign)
async def modify_workflow(request: ModifyWorkflowRequest) -> AgentSystemDesign:
    """
    Internal API: Modify workflow (HITL)

    Used by runtime component during human-in-the-loop
    """
    try:
        result = await workflow_service.modify_agent_system(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/version/bump", response_model=VersionWorkflowResponse)
def bump_version(request: VersionWorkflowRequest) -> VersionWorkflowResponse:
    """
    Internal API: Bump workflow version

    Used by storage component during final save
    """
    try:
        result = workflow_service.bump_workflow_version(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
