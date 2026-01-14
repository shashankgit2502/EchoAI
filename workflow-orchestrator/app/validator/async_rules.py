"""
Asynchronous Validation Rules
MCP tool availability and LLM connectivity checks (requires external calls)
"""
from typing import List
import asyncio

from app.schemas.api_models import AgentSystemDesign
from app.validator.errors import (
    ValidationError,
    configuration_warning,
    ValidationErrorType
)
from app.validator.retry import with_retry_and_timeout, RateLimiter
from app.core.constants import ValidationSeverity
from app.core.logging import get_logger

logger = get_logger(__name__)

# Rate limiter for external calls
_rate_limiter = RateLimiter(calls_per_second=5.0)


# ============================================================================
# MCP TOOL AVAILABILITY
# ============================================================================

@with_retry_and_timeout(max_attempts=2, timeout_seconds=10)
async def check_mcp_tool_availability(tool_name: str, tool_config: dict) -> bool:
    """
    Check if MCP tool is available

    Phase 3: Will be implemented with MCP client
    Phase 4: Placeholder that returns True
    """
    await _rate_limiter.acquire()

    logger.debug(f"Checking MCP tool availability: {tool_name}")

    # Phase 3 TODO: Implement actual MCP health check
    # from app.tools.mcp_client import MCPClient
    # client = MCPClient()
    # return await client.check_tool_availability(tool_name, tool_config)

    # Placeholder: Assume tools are available
    return True


async def validate_mcp_tools(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate all MCP tools are accessible

    Phase 3: Full implementation
    Phase 4: Placeholder with warnings
    """
    warnings = []

    mcp_tools = [tool for tool in system.tools if tool.type == "mcp"]

    if not mcp_tools:
        logger.debug("No MCP tools to validate")
        return warnings

    logger.info(f"Validating {len(mcp_tools)} MCP tools...")

    # Check each MCP tool (Phase 3 will implement this)
    for tool in mcp_tools:
        try:
            # Phase 3 TODO: Actual availability check
            is_available = await check_mcp_tool_availability(tool.name, tool.config)

            if not is_available:
                warnings.append(ValidationError(
                    severity=ValidationSeverity.WARNING,
                    error_type=ValidationErrorType.MCP_ERROR,
                    location=f"tools.{tool.name}",
                    message=f"MCP tool '{tool.name}' may not be available",
                    code="W5001",
                    suggestion="Verify MCP server is running and tool is registered"
                ))

        except Exception as e:
            logger.warning(f"Failed to check MCP tool '{tool.name}': {e}")
            warnings.append(ValidationError(
                severity=ValidationSeverity.WARNING,
                error_type=ValidationErrorType.MCP_ERROR,
                location=f"tools.{tool.name}",
                message=f"Could not verify MCP tool availability: {str(e)}",
                code="W5002",
                suggestion="Check MCP server connectivity"
            ))

    return warnings


# ============================================================================
# LLM AVAILABILITY
# ============================================================================

@with_retry_and_timeout(max_attempts=2, timeout_seconds=15)
async def check_llm_availability(model_name: str) -> bool:
    """
    Check if LLM model is accessible

    Makes a minimal API call to verify connectivity

    Phase 6: Full implementation with LLM provider ✅
    """
    await _rate_limiter.acquire()

    logger.debug(f"Checking LLM availability: {model_name}")

    # Use LLM provider to check model availability
    from app.services.llm_provider import get_llm_provider

    try:
        provider = get_llm_provider()
        return await provider.check_availability(model_name)

    except Exception as e:
        logger.warning(f"Failed to check LLM availability for {model_name}: {e}")
        # Don't fail validation on availability check failures
        # Just log and return False
        return False


async def validate_llm_availability(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate all LLM models are accessible

    Phase 4: Placeholder with warnings
    Phase 6: Full implementation
    """
    warnings = []

    # Get unique models used
    models_used = set(agent.llm_config.model for agent in system.agents)

    logger.info(f"Validating {len(models_used)} unique LLM models...")

    for model in models_used:
        try:
            # Phase 6 TODO: Actual availability check
            is_available = await check_llm_availability(model)

            if not is_available:
                warnings.append(ValidationError(
                    severity=ValidationSeverity.WARNING,
                    error_type=ValidationErrorType.LLM_ERROR,
                    location="llm_config",
                    message=f"LLM model '{model}' may not be accessible",
                    code="W4001",
                    suggestion="Verify API keys and model access permissions"
                ))

        except Exception as e:
            logger.warning(f"Failed to check LLM model '{model}': {e}")
            warnings.append(ValidationError(
                severity=ValidationSeverity.WARNING,
                error_type=ValidationErrorType.LLM_ERROR,
                location="llm_config",
                message=f"Could not verify LLM availability: {str(e)}",
                code="W4002",
                suggestion="Check API keys and network connectivity"
            ))

    return warnings


# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

async def validate_external_dependencies(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Validate all external dependencies (MCP tools, LLMs, APIs)

    This is a comprehensive check that runs all async validations
    """
    logger.info("Validating external dependencies...")

    # Run all async validations concurrently
    results = await asyncio.gather(
        validate_mcp_tools(system),
        validate_llm_availability(system),
        return_exceptions=True
    )

    # Collect all errors
    all_errors: List[ValidationError] = []

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Async validation failed: {result}")
            all_errors.append(ValidationError(
                severity=ValidationSeverity.WARNING,
                error_type=ValidationErrorType.DEPENDENCY_ERROR,
                location="external_dependencies",
                message=f"Dependency check failed: {str(result)}",
                code="W3001",
                suggestion="Check system connectivity and external service availability"
            ))
        elif isinstance(result, list):
            all_errors.extend(result)

    logger.info(f"External dependency validation complete: {len(all_errors)} issues found")

    return all_errors


# ============================================================================
# MAIN ASYNC VALIDATION
# ============================================================================

async def run_all_async_validations(system: AgentSystemDesign) -> List[ValidationError]:
    """
    Run all asynchronous validation rules

    This is the main entry point for async validation
    """
    logger.info(f"Running asynchronous validation for: {system.system_name}")

    try:
        errors = await validate_external_dependencies(system)
        logger.info(f"Asynchronous validation complete: {len(errors)} issues found")
        return errors

    except Exception as e:
        logger.error(f"Async validation failed critically: {e}", exc_info=True)
        return [ValidationError(
            severity=ValidationSeverity.WARNING,
            error_type=ValidationErrorType.DEPENDENCY_ERROR,
            location="async_validation",
            message=f"Async validation failed: {str(e)}",
            code="W3002",
            suggestion="Review logs for detailed error information"
        )]
