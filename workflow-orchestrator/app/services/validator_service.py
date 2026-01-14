"""
Validator Service
Service layer wrapper for validation logic
Provides clean interface for internal API communication
"""
from typing import Optional
from app.schemas.api_models import (
    ValidateAgentSystemRequest,
    ValidateAgentRequest,
    ValidateWorkflowRequest,
    ValidationResponse,
    ValidationError as APIValidationError,
    AgentSystemDesign,
    AgentDefinition,
    WorkflowDefinition
)
from app.validator.validator import AgentSystemValidator
from app.validator.errors import ValidationError, ValidationResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class ValidatorService:
    """
    Service layer for validation operations

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Async + idempotent
    - Ready for microservice extraction
    """

    def __init__(self, enable_async_validation: bool = True):
        """Initialize validator service"""
        self._validator = AgentSystemValidator(enable_async_validation)
        logger.info("ValidatorService initialized")

    async def validate_agent_system(
        self,
        request: ValidateAgentSystemRequest
    ) -> ValidationResponse:
        """
        Validate complete agent system

        Args:
            request: Validation request with agent system

        Returns:
            ValidationResponse with errors/warnings
        """
        logger.info(f"Validating agent system: {request.agent_system.system_name}")

        # Skip async validation for draft mode (faster feedback)
        skip_async = request.mode == "draft"

        # Call core validator
        result = await self._validator.validate(
            request.agent_system,
            skip_async=skip_async
        )

        # Convert to API response format
        return self._convert_validation_result(result)

    async def validate_agent(
        self,
        request: ValidateAgentRequest
    ) -> ValidationResponse:
        """
        Validate single agent definition

        Args:
            request: Agent validation request

        Returns:
            ValidationResponse with errors/warnings
        """
        logger.info(f"Validating agent: {request.agent.id}")

        # Create minimal agent system for validation
        temp_system = AgentSystemDesign(
            system_name="temp_validation",
            description="Temporary system for agent validation",
            domain="validation",
            agents=[request.agent],
            workflows=[],
            communication_pattern="sequential"
        )

        result = await self._validator.validate(temp_system, skip_async=True)
        return self._convert_validation_result(result)

    async def validate_workflow(
        self,
        request: ValidateWorkflowRequest
    ) -> ValidationResponse:
        """
        Validate single workflow definition

        Args:
            request: Workflow validation request

        Returns:
            ValidationResponse with errors/warnings
        """
        logger.info(f"Validating workflow: {request.workflow.name}")

        # Create minimal agent system for validation
        temp_system = AgentSystemDesign(
            system_name="temp_validation",
            description="Temporary system for workflow validation",
            domain="validation",
            agents=[],
            workflows=[request.workflow],
            communication_pattern="sequential"
        )

        result = await self._validator.validate(temp_system, skip_async=True)
        return self._convert_validation_result(result)

    async def quick_validate(
        self,
        agent_system: AgentSystemDesign
    ) -> bool:
        """
        Quick validation (sync only)

        Args:
            agent_system: System to validate

        Returns:
            True if valid, False otherwise
        """
        result = await self._validator.validate(agent_system, skip_async=True)
        return result.valid

    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================

    def _convert_validation_result(self, result: ValidationResult) -> ValidationResponse:
        """Convert internal ValidationResult to API ValidationResponse"""

        errors = []
        warnings = []

        for error in result.errors:
            api_error = APIValidationError(
                severity=error.severity.value,
                location=error.location,
                message=error.message,
                suggestion=error.suggestion
            )

            if error.severity.value == "error":
                errors.append(api_error)
            elif error.severity.value == "warning":
                warnings.append(api_error)

        return ValidationResponse(
            valid=result.valid,
            errors=errors,
            warnings=warnings
        )


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_validator_service: Optional[ValidatorService] = None


def get_validator_service(enable_async: bool = True) -> ValidatorService:
    """
    Get singleton validator service instance

    Args:
        enable_async: Enable async validation

    Returns:
        ValidatorService instance
    """
    global _validator_service

    if _validator_service is None:
        _validator_service = ValidatorService(enable_async)

    return _validator_service
