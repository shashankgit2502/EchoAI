"""
External Workflow API Routes
User-facing workflow design and management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
from app.schemas.api_models import (
    UserRequest,
    DesignWorkflowResponse,
    DomainAnalysis,
    AgentSystemDesign,
    SaveWorkflowRequest,
    SaveWorkflowResponse,
    CloneWorkflowRequest,
    LoadWorkflowRequest,
    ModifyWorkflowRequest,
    VersionWorkflowRequest,
    VersionWorkflowResponse,
    ProcessMessageRequest,
    ProcessMessageResponse
)
from app.services.workflow_service import get_workflow_service
from app.services.storage_service import get_storage_service
from app.services.validator_service import get_validator_service
from app.core.logging import get_logger

router = APIRouter(prefix="/workflow", tags=["Workflow"])
logger = get_logger(__name__)

# Get service instances
workflow_service = get_workflow_service()
storage_service = get_storage_service()
validator_service = get_validator_service()


@router.post("/design", response_model=DesignWorkflowResponse)
async def design_workflow(user_request: UserRequest) -> DesignWorkflowResponse:
    """
    Design workflow from natural language request

    Complete pipeline:
    1. Analyze user request
    2. Generate meta-prompt
    3. Design agent system with LLM

    Required payload:
    {
        "request": "Your workflow description (minimum 10 characters)",
        "context": {}  // optional
    }
    """
    try:
        logger.info(f"Received workflow design request: {user_request.request[:100]}")

        agent_system, analysis, meta_prompt = await workflow_service.design_from_user_request(
            user_request
        )

        logger.info(f"Workflow design completed successfully")
        return DesignWorkflowResponse(
            analysis=analysis,
            agent_system=agent_system,
            meta_prompt_used=meta_prompt
        )

    except Exception as e:
        logger.error(f"Workflow design failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow design failed: {str(e)}"
        )


@router.post("/modify", response_model=AgentSystemDesign)
async def modify_workflow(request: ModifyWorkflowRequest) -> AgentSystemDesign:
    """
    Modify existing workflow (HITL)

    Allows human-in-the-loop modifications
    """
    try:
        modified_system = await workflow_service.modify_agent_system(request)
        return modified_system

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Modification failed: {str(e)}"
        )


@router.post("/save/draft", response_model=SaveWorkflowResponse)
def save_draft(request: SaveWorkflowRequest) -> SaveWorkflowResponse:
    """Save workflow as DRAFT"""
    try:
        return storage_service.save_draft(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save/temp", response_model=SaveWorkflowResponse)
async def save_temp(request: SaveWorkflowRequest) -> SaveWorkflowResponse:
    """
    Save workflow as TEMP (validated, for testing)
    """
    try:
        # Validate before saving as TEMP
        from app.schemas.api_models import ValidateAgentSystemRequest
        validation = await validator_service.validate_agent_system(
            ValidateAgentSystemRequest(agent_system=request.agent_system, mode="temp")
        )

        if not validation.valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Validation failed", "errors": [e.dict() for e in validation.errors]}
            )

        return storage_service.save_temp(request)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save/final", response_model=SaveWorkflowResponse)
async def save_final(request: SaveWorkflowRequest) -> SaveWorkflowResponse:
    """
    Save workflow as FINAL (immutable, versioned)
    """
    try:
        # Validate before saving as FINAL
        from app.schemas.api_models import ValidateAgentSystemRequest
        validation = await validator_service.validate_agent_system(
            ValidateAgentSystemRequest(agent_system=request.agent_system, mode="final")
        )

        if not validation.valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"message": "Validation failed", "errors": [e.dict() for e in validation.errors]}
            )

        return storage_service.save_final(request)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load", response_model=AgentSystemDesign)
def load_workflow(request: LoadWorkflowRequest) -> AgentSystemDesign:
    """Load workflow from storage"""
    try:
        workflow = storage_service.load_workflow(request)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clone", response_model=AgentSystemDesign)
def clone_workflow(request: CloneWorkflowRequest) -> AgentSystemDesign:
    """Clone FINAL workflow to DRAFT for editing"""
    try:
        return storage_service.clone_final_to_draft(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions/{workflow_id}", response_model=List[str])
def list_versions(workflow_id: str) -> List[str]:
    """List all versions of a workflow"""
    try:
        return storage_service.list_versions(workflow_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/version/bump", response_model=VersionWorkflowResponse)
def bump_version(request: VersionWorkflowRequest) -> VersionWorkflowResponse:
    """Bump workflow version"""
    try:
        return workflow_service.bump_workflow_version(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/draft/{workflow_id}")
def delete_draft(workflow_id: str) -> dict:
    """Delete DRAFT workflow"""
    try:
        from app.schemas.api_models import DeleteWorkflowRequest
        request = DeleteWorkflowRequest(workflow_id=workflow_id, state="draft")
        success = storage_service.delete_workflow(request)

        if not success:
            raise HTTPException(status_code=404, detail="Draft not found")

        return {"success": True, "message": f"Draft deleted: {workflow_id}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/temp/{workflow_id}")
def delete_temp(workflow_id: str) -> dict:
    """Delete TEMP workflow"""
    try:
        from app.schemas.api_models import DeleteWorkflowRequest
        request = DeleteWorkflowRequest(workflow_id=workflow_id, state="temp")
        success = storage_service.delete_workflow(request)

        if not success:
            raise HTTPException(status_code=404, detail="Temp not found")

        return {"success": True, "message": f"Temp deleted: {workflow_id}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/archive/{workflow_id}")
def archive_version(workflow_id: str, version: str) -> dict:
    """
    Archive a FINAL version

    Note: Cannot delete FINAL versions, only archive them
    """
    try:
        from app.schemas.api_models import ArchiveWorkflowRequest
        request = ArchiveWorkflowRequest(workflow_id=workflow_id, version=version)
        success = storage_service.archive_version(request)

        if not success:
            raise HTTPException(status_code=404, detail="Version not found")

        return {"success": True, "message": f"Version archived: {workflow_id} v{version}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
def list_workflows(state: str = None) -> dict:
    """
    List all workflows

    Optional state filter: draft, temp, final
    """
    try:
        result = storage_service.list_workflows(state)
        return result.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UNIFIED PROCESS ENDPOINT (NEW - does not modify existing endpoints)
# ============================================================================

@router.post("/process", response_model=ProcessMessageResponse)
async def process_message(request: ProcessMessageRequest) -> ProcessMessageResponse:
    """
    Unified message processing endpoint.

    Backend handles intent detection and routes to appropriate action:
    - generate: Create new workflow
    - modify: Modify existing workflow
    - test: Test workflow
    - execute: Execute workflow with user input
    - save: Save workflow as final

    This is the recommended endpoint for frontend to use.
    All existing endpoints continue to work as before.
    """
    from app.services.process_service import process_service

    try:
        logger.info(f"Process endpoint called: {request.message[:50]}...")
        result = await process_service.process_message(request)
        return result

    except Exception as e:
        logger.error(f"Process endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )
