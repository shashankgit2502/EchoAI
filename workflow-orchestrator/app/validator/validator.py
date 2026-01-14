"""
Agent System Validator - Main Orchestrator
Coordinates synchronous and asynchronous validation
"""
from typing import Optional
from datetime import datetime

from app.schemas.api_models import AgentSystemDesign
from app.validator.errors import ValidationResult, ValidationError
from app.validator.sync_rules import run_all_sync_validations
from app.validator.async_rules import run_all_async_validations
from app.core.logging import get_logger
from app.core.telemetry import trace_operation
from app.core.constants import TelemetrySpan, TelemetryAttribute

logger = get_logger(__name__)


# ============================================================================
# MAIN VALIDATOR
# ============================================================================

class AgentSystemValidator:
    """
    Complete validation pipeline orchestrator

    Coordinates:
    1. Synchronous validation (schema, topology, references)
    2. Asynchronous validation (MCP tools, LLM availability)
    3. Error collection and reporting

    Usage:
        validator = AgentSystemValidator()
        result = await validator.validate(agent_system)

        if result.valid:
            # proceed with workflow
        else:
            # handle validation errors
    """

    def __init__(self, enable_async_validation: bool = True):
        """
        Initialize validator

        Args:
            enable_async_validation: Enable async checks (MCP, LLM)
                                    Set to False for faster validation during development
        """
        self.enable_async_validation = enable_async_validation
        logger.info(f"Validator initialized (async={enable_async_validation})")

    async def validate(
        self,
        agent_system: AgentSystemDesign,
        skip_async: bool = False
    ) -> ValidationResult:
        """
        Complete validation pipeline

        Args:
            agent_system: The agent system to validate
            skip_async: Skip async validation (for faster validation)

        Returns:
            ValidationResult with all errors, warnings, and info
        """
        with trace_operation(
            TelemetrySpan.WORKFLOW_VALIDATE,
            {
                TelemetryAttribute.WORKFLOW_NAME: agent_system.system_name,
                "agent_count": len(agent_system.agents),
                "workflow_count": len(agent_system.workflows)
            }
        ):
            logger.info(f"Starting validation: {agent_system.system_name}")

            all_errors: list[ValidationError] = []

            # Step 1: Synchronous validation (always runs)
            logger.info("Running synchronous validation...")
            sync_errors = run_all_sync_validations(agent_system)
            all_errors.extend(sync_errors)

            logger.info(f"Sync validation complete: {len(sync_errors)} issues")

            # Step 2: Asynchronous validation (optional)
            if self.enable_async_validation and not skip_async:
                logger.info("Running asynchronous validation...")

                try:
                    async_errors = await run_all_async_validations(agent_system)
                    all_errors.extend(async_errors)
                    logger.info(f"Async validation complete: {len(async_errors)} issues")

                except Exception as e:
                    logger.error(f"Async validation failed: {e}", exc_info=True)
                    # Don't block on async validation failures
                    # Add a warning instead
                    all_errors.append(ValidationError(
                        severity="warning",
                        error_type="dependency_error",
                        location="async_validation",
                        message=f"Async validation failed: {str(e)}",
                        code="W9999",
                        suggestion="Check logs for details. Proceeding without async validation."
                    ))
            else:
                logger.info("Skipping asynchronous validation")

            # Build result
            result = ValidationResult.create_invalid(all_errors)

            logger.info(
                f"Validation complete: valid={result.valid}, "
                f"errors={result.error_count}, "
                f"warnings={result.warning_count}"
            )

            return result

    async def validate_quick(self, agent_system: AgentSystemDesign) -> ValidationResult:
        """
        Quick validation (sync only, no async checks)

        Use this for rapid iteration during development

        Args:
            agent_system: The agent system to validate

        Returns:
            ValidationResult (sync checks only)
        """
        return await self.validate(agent_system, skip_async=True)

    def validate_sync_only(self, agent_system: AgentSystemDesign) -> ValidationResult:
        """
        Synchronous validation only (no async/await)

        Use this when you need synchronous validation

        Args:
            agent_system: The agent system to validate

        Returns:
            ValidationResult (sync checks only)
        """
        logger.info(f"Running sync-only validation: {agent_system.system_name}")

        errors = run_all_sync_validations(agent_system)
        result = ValidationResult.create_invalid(errors)

        logger.info(f"Sync validation complete: valid={result.valid}")

        return result


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

async def validate_agent_system(
    agent_system: AgentSystemDesign,
    quick: bool = False
) -> ValidationResult:
    """
    Convenience function for validation

    Args:
        agent_system: The agent system to validate
        quick: If True, skip async validation

    Returns:
        ValidationResult
    """
    validator = AgentSystemValidator()

    if quick:
        return await validator.validate_quick(agent_system)
    else:
        return await validator.validate(agent_system)


def validate_agent_system_sync(agent_system: AgentSystemDesign) -> ValidationResult:
    """
    Synchronous validation (no async)

    Args:
        agent_system: The agent system to validate

    Returns:
        ValidationResult
    """
    validator = AgentSystemValidator(enable_async_validation=False)
    return validator.validate_sync_only(agent_system)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# For backward compatibility with existing code
from app.schemas.api_models import ValidationResponse


async def validate_for_api(agent_system: AgentSystemDesign) -> ValidationResponse:
    """
    Validate and return API-compatible response

    This converts the new ValidationResult to the old ValidationResponse format
    for backward compatibility

    Args:
        agent_system: The agent system to validate

    Returns:
        ValidationResponse (old format)
    """
    validator = AgentSystemValidator()
    result = await validator.validate(agent_system)

    # Convert to old format
    from app.schemas.api_models import ValidationError as ApiValidationError

    errors_list = [
        ApiValidationError(
            severity=e.severity.value,
            location=e.location,
            message=e.message,
            suggestion=e.suggestion
        )
        for e in result.errors
    ]

    warnings_list = [
        ApiValidationError(
            severity=w.severity.value,
            location=w.location,
            message=w.message,
            suggestion=w.suggestion
        )
        for w in result.warnings
    ]

    return ValidationResponse(
        valid=result.valid,
        errors=errors_list,
        warnings=warnings_list,
        validated_at=datetime.utcnow()
    )
