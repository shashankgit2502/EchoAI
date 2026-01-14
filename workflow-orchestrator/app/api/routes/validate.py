"""
External Validation API Routes
User-facing validation endpoints
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas.api_models import (
    AgentSystemDesign,
    ValidationResponse,
    ValidateAgentSystemRequest
)
from app.services.validator_service import get_validator_service

router = APIRouter(prefix="/validate", tags=["Validation"])

# Get service instance
validator_service = get_validator_service()


@router.post("", response_model=ValidationResponse)
@router.post("/", response_model=ValidationResponse)
async def validate_agent_system(agent_system: AgentSystemDesign) -> ValidationResponse:
    """
    Validate an agent system design

    Checks:
    - Agent uniqueness
    - Tool references
    - Workflow references
    - Communication patterns
    - LLM configurations
    - System prompts
    - Hierarchical structure (if applicable)
    """
    try:
        request = ValidateAgentSystemRequest(
            agent_system=agent_system,
            mode="final"  # Full validation for external API
        )

        result = await validator_service.validate_agent_system(request)

        if not result.valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Validation failed",
                    "errors": [e.dict() for e in result.errors],
                    "warnings": [w.dict() for w in result.warnings]
                }
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@router.post("/draft", response_model=ValidationResponse)
async def validate_draft(request: ValidateAgentSystemRequest) -> ValidationResponse:
    """
    Validate draft agent system (sync validation only, faster)

    Used after initial workflow design, before HITL
    """
    try:
        # Override mode to draft
        request.mode = "draft"
        result = await validator_service.validate_agent_system(request)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Draft validation failed: {str(e)}"
        )


@router.post("/final", response_model=ValidationResponse)
async def validate_final(request: ValidateAgentSystemRequest) -> ValidationResponse:
    """
    Validate final agent system (full validation including async checks)

    Used after HITL, before saving as temp/final
    """
    try:
        # Override mode to final
        request.mode = "final"
        result = await validator_service.validate_agent_system(request)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Final validation failed: {str(e)}"
        )


@router.post("/quick", response_model=dict)
async def quick_validate(agent_system: AgentSystemDesign) -> dict:
    """
    Quick validation (sync only, no async checks)

    Faster validation for draft mode
    """
    try:
        is_valid = await validator_service.quick_validate(agent_system)

        return {
            "valid": is_valid,
            "mode": "quick"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick validation failed: {str(e)}"
        )
