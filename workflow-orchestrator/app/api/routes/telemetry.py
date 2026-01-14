"""
External Telemetry API Routes
User-facing metrics and observability endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.api_models import (
    TelemetryQuery,
    TelemetryResponse
)
from app.services.telemetry_service import get_telemetry_service

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

# Get service instance
telemetry_service = get_telemetry_service()


@router.post("/metrics", response_model=TelemetryResponse)
async def query_metrics(query: TelemetryQuery) -> TelemetryResponse:
    """
    Query execution metrics

    Supports filtering by:
    - workflow_id
    - run_id
    - agent_id
    - time range
    """
    try:
        result = await telemetry_service.query_metrics(query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metrics query failed: {str(e)}"
        )


@router.get("/workflow/{workflow_id}/history")
async def get_workflow_history(workflow_id: str, limit: int = 100) -> List[dict]:
    """
    Get execution history for workflow
    """
    try:
        history = await telemetry_service.get_workflow_history(workflow_id, limit)
        return history
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"History retrieval failed: {str(e)}"
        )


@router.get("/cost/{run_id}")
async def get_cost_breakdown(run_id: str) -> dict:
    """
    Get cost breakdown for execution (in INR)
    """
    try:
        breakdown = await telemetry_service.get_cost_breakdown(run_id)
        return breakdown
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cost breakdown failed: {str(e)}"
        )
