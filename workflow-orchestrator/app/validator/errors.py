"""
Validation Error Types
Structured error handling for validation pipeline
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from app.core.constants import ValidationSeverity, ErrorCode


# ============================================================================
# ERROR TYPES
# ============================================================================

class ValidationErrorType(str, Enum):
    """Types of validation errors"""
    SCHEMA_ERROR = "schema_error"
    TOPOLOGY_ERROR = "topology_error"
    REFERENCE_ERROR = "reference_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    LIMIT_ERROR = "limit_error"
    MCP_ERROR = "mcp_error"
    LLM_ERROR = "llm_error"


# ============================================================================
# STRUCTURED VALIDATION ERROR
# ============================================================================

@dataclass
class ValidationError:
    """
    Structured validation error

    This is the core error type used throughout validation.
    Unlike Exception, this is a data structure that can be collected and reported.
    """
    severity: ValidationSeverity
    error_type: ValidationErrorType
    location: str
    message: str
    code: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "severity": self.severity.value,
            "error_type": self.error_type.value,
            "location": self.location,
            "message": self.message,
            "code": self.code,
            "suggestion": self.suggestion,
            "details": self.details
        }

    def is_blocking(self) -> bool:
        """Check if this error blocks progression"""
        return self.severity == ValidationSeverity.ERROR

    def is_warning(self) -> bool:
        """Check if this is a warning"""
        return self.severity == ValidationSeverity.WARNING


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class ValidationResult:
    """
    Complete validation result

    Contains all errors, warnings, and info messages from validation
    """
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]

    @property
    def has_errors(self) -> bool:
        """Check if there are any blocking errors"""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0

    @property
    def error_count(self) -> int:
        """Total error count"""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Total warning count"""
        return len(self.warnings)

    @property
    def total_issues(self) -> int:
        """Total issues (errors + warnings + info)"""
        return len(self.errors) + len(self.warnings) + len(self.info)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "total_issues": self.total_issues,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info]
        }

    @classmethod
    def create_valid(cls) -> "ValidationResult":
        """Create a valid result with no errors"""
        return cls(valid=True, errors=[], warnings=[], info=[])

    @classmethod
    def create_invalid(cls, errors: List[ValidationError]) -> "ValidationResult":
        """Create an invalid result with errors"""
        warnings = [e for e in errors if e.is_warning()]
        blocking_errors = [e for e in errors if e.is_blocking()]
        info_messages = [e for e in errors if e.severity == ValidationSeverity.INFO]

        return cls(
            valid=len(blocking_errors) == 0,
            errors=blocking_errors,
            warnings=warnings,
            info=info_messages
        )


# ============================================================================
# EXCEPTION TYPES (for critical failures)
# ============================================================================

class ValidationException(Exception):
    """
    Base exception for critical validation failures

    Use this for catastrophic failures where validation cannot continue.
    For expected validation errors, use ValidationError dataclass instead.
    """
    pass


class SchemaValidationException(ValidationException):
    """Schema validation failed critically"""
    pass


class TopologyValidationException(ValidationException):
    """Topology validation failed critically"""
    pass


class DependencyValidationException(ValidationException):
    """Dependency validation failed critically"""
    pass


class TimeoutException(ValidationException):
    """Validation operation timed out"""
    pass


class RetryExhaustedException(ValidationException):
    """Retry attempts exhausted"""
    pass


# ============================================================================
# ERROR BUILDERS (convenience functions)
# ============================================================================

def schema_error(
    location: str,
    message: str,
    suggestion: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> ValidationError:
    """Build a schema validation error"""
    return ValidationError(
        severity=ValidationSeverity.ERROR,
        error_type=ValidationErrorType.SCHEMA_ERROR,
        location=location,
        message=message,
        code=ErrorCode.SCHEMA_VALIDATION_ERROR,
        suggestion=suggestion,
        details=details
    )


def topology_error(
    location: str,
    message: str,
    suggestion: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> ValidationError:
    """Build a topology validation error"""
    return ValidationError(
        severity=ValidationSeverity.ERROR,
        error_type=ValidationErrorType.TOPOLOGY_ERROR,
        location=location,
        message=message,
        code=ErrorCode.TOPOLOGY_VALIDATION_ERROR,
        suggestion=suggestion,
        details=details
    )


def reference_error(
    location: str,
    message: str,
    missing_ref: str,
    suggestion: Optional[str] = None
) -> ValidationError:
    """Build a reference validation error"""
    return ValidationError(
        severity=ValidationSeverity.ERROR,
        error_type=ValidationErrorType.REFERENCE_ERROR,
        location=location,
        message=message,
        code=ErrorCode.TOOL_REFERENCE_ERROR if "tool" in location.lower() else ErrorCode.AGENT_REFERENCE_ERROR,
        suggestion=suggestion,
        details={"missing_reference": missing_ref}
    )


def configuration_warning(
    location: str,
    message: str,
    suggestion: Optional[str] = None
) -> ValidationError:
    """Build a configuration warning"""
    return ValidationError(
        severity=ValidationSeverity.WARNING,
        error_type=ValidationErrorType.CONFIGURATION_ERROR,
        location=location,
        message=message,
        code="W1001",
        suggestion=suggestion
    )


def info_message(
    location: str,
    message: str,
    suggestion: Optional[str] = None
) -> ValidationError:
    """Build an info message"""
    return ValidationError(
        severity=ValidationSeverity.INFO,
        error_type=ValidationErrorType.CONFIGURATION_ERROR,
        location=location,
        message=message,
        code="I1001",
        suggestion=suggestion
    )
