"""
External Runtime API Routes
User-facing execution endpoints
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List
import json
from app.schemas.api_models import (
    ExecuteWorkflowRequest,
    ExecutionStatus,
    RuntimeExecuteRequest,
    RuntimeResumeRequest
)
from app.services.runtime_service import get_runtime_service

router = APIRouter(prefix="/runtime", tags=["Runtime"])

# Get service instance
runtime_service = get_runtime_service()


@router.post("/execute", response_model=ExecutionStatus)
async def execute_workflow(request: ExecuteWorkflowRequest) -> ExecutionStatus:
    """
    Execute a workflow

    Modes:
    - test: Execute TEMP workflow (for testing)
    - final: Execute FINAL workflow (production)

    Optional thread_id enables HITL support (checkpointing)
    """
    try:
        # Convert to internal request format
        runtime_request = RuntimeExecuteRequest(
            workflow_id=request.workflow_id,
            execution_mode=request.execution_mode,
            version=request.version,
            input_payload=request.input_payload,
            thread_id=request.thread_id,
            enable_hitl=True
        )

        result = await runtime_service.execute_workflow(runtime_request)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


@router.post("/resume", response_model=ExecutionStatus)
async def resume_execution(
    workflow_id: str,
    thread_id: str,
    human_decision: dict
) -> ExecutionStatus:
    """
    Resume paused execution with human input (HITL)

    Requires thread_id from original execution
    """
    try:
        request = RuntimeResumeRequest(
            workflow_id=workflow_id,
            thread_id=thread_id,
            human_decision=human_decision
        )

        result = await runtime_service.resume_execution(request)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume failed: {str(e)}"
        )


@router.get("/status/{run_id}", response_model=ExecutionStatus)
def get_execution_status(run_id: str) -> ExecutionStatus:
    """Get execution status by run ID"""
    try:
        status = runtime_service.get_execution_status(run_id)
        if not status:
            raise HTTPException(status_code=404, detail="Execution not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cancel/{run_id}")
async def cancel_execution(run_id: str) -> dict:
    """Cancel running execution"""
    try:
        success = await runtime_service.cancel_execution(run_id)

        if not success:
            raise HTTPException(status_code=404, detail="Execution not found")

        return {"success": True, "run_id": run_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute/stream")
async def execute_workflow_streaming(request: ExecuteWorkflowRequest):
    """
    Execute workflow with streaming updates

    Returns Server-Sent Events (SSE) with execution progress
    """
    async def generate():
        try:
            # Convert to internal request format
            runtime_request = RuntimeExecuteRequest(
                workflow_id=request.workflow_id,
                execution_mode=request.execution_mode,
                version=request.version,
                input_payload=request.input_payload,
                thread_id=request.thread_id,
                enable_hitl=True
            )

            # Stream execution updates
            async for status in runtime_service.execute_streaming(runtime_request):
                # Format as SSE
                data = json.dumps(status.dict(), default=str)
                yield f"data: {data}\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.get("/active", response_model=List[ExecutionStatus])
def list_active_executions() -> List[ExecutionStatus]:
    """
    List all active executions
    """
    try:
        return runtime_service.list_active_executions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[ExecutionStatus])
async def execute_batch(requests: List[ExecuteWorkflowRequest]) -> List[ExecutionStatus]:
    """
    Execute multiple workflows in parallel
    """
    try:
        # Convert to internal request format
        runtime_requests = [
            RuntimeExecuteRequest(
                workflow_id=req.workflow_id,
                execution_mode=req.execution_mode,
                version=req.version,
                input_payload=req.input_payload,
                thread_id=req.thread_id,
                enable_hitl=True
            )
            for req in requests
        ]

        results = await runtime_service.execute_batch(runtime_requests)
        return results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch execution failed: {str(e)}"
        )


@router.get("/history", response_model=List[ExecutionStatus])
def get_execution_history(
    workflow_id: str = None,
    limit: int = 100
) -> List[ExecutionStatus]:
    """
    Get execution history

    Optional workflow_id filter
    """
    try:
        return runtime_service.get_execution_history(workflow_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
