"""
Internal Runtime API
Component-to-component execution operations
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from app.schemas.api_models import (
    RuntimeExecuteRequest,
    RuntimeResumeRequest,
    ExecutionStatus,
    RuntimeMetrics,
    CheckpointInfo
)
from app.services.runtime_service import get_runtime_service

router = APIRouter(prefix="/api/internal/runtime", tags=["Internal-Runtime"])

# Get service instance
runtime_service = get_runtime_service()


@router.post("/execute", response_model=ExecutionStatus)
async def execute_workflow(request: RuntimeExecuteRequest) -> ExecutionStatus:
    """
    Internal API: Execute workflow

    Used by external API or other components
    """
    try:
        result = await runtime_service.execute_workflow(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume", response_model=ExecutionStatus)
async def resume_execution(request: RuntimeResumeRequest) -> ExecutionStatus:
    """
    Internal API: Resume paused execution (HITL)
    """
    try:
        result = await runtime_service.resume_execution(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{run_id}", response_model=Optional[RuntimeMetrics])
def get_metrics(run_id: str) -> Optional[RuntimeMetrics]:
    """
    Internal API: Get execution metrics
    """
    try:
        result = runtime_service.get_execution_metrics(run_id)
        if not result:
            raise HTTPException(status_code=404, detail="Metrics not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoint/{thread_id}", response_model=Optional[CheckpointInfo])
def get_checkpoint(thread_id: str) -> Optional[CheckpointInfo]:
    """
    Internal API: Get checkpoint for thread
    """
    try:
        result = runtime_service.get_checkpoint(thread_id)
        if not result:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
