"""
Internal Validator API
Component-to-component validation endpoints
"""
from fastapi import APIRouter, HTTPException
from app.schemas.api_models import (
    ValidateAgentSystemRequest,
    ValidateAgentRequest,
    ValidateWorkflowRequest,
    ValidationResponse
)
from app.services.validator_service import get_validator_service

router = APIRouter(prefix="/api/internal/validator", tags=["Internal-Validator"])

# Get service instance
validator_service = get_validator_service()


@router.post("/agent-system", response_model=ValidationResponse)
async def validate_agent_system(request: ValidateAgentSystemRequest) -> ValidationResponse:
    """
    Internal API: Validate agent system

    Used by other components (workflow, runtime, storage)
    """
    try:
        result = await validator_service.validate_agent_system(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent", response_model=ValidationResponse)
async def validate_agent(request: ValidateAgentRequest) -> ValidationResponse:
    """
    Internal API: Validate single agent
    """
    try:
        result = await validator_service.validate_agent(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow", response_model=ValidationResponse)
async def validate_workflow(request: ValidateWorkflowRequest) -> ValidationResponse:
    """
    Internal API: Validate workflow
    """
    try:
        result = await validator_service.validate_workflow(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
