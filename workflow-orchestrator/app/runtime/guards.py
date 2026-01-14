"""
Runtime Guards
Enforce cost, timeout, and step limits during workflow execution
Prevents runaway executions and excessive resource consumption
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.core.constants import SystemLimits
from app.core.logging import get_logger
from app.utils.time import utc_now, duration_seconds
from app.services.llm_provider import get_llm_provider

logger = get_logger(__name__)


# ============================================================================
# GUARD VIOLATIONS
# ============================================================================

@dataclass
class GuardViolation:
    """
    Represents a guard violation

    Attributes:
        guard_type: Type of guard violated
        message: Violation message
        current_value: Current measured value
        limit_value: Configured limit value
        timestamp: When violation occurred
    """
    guard_type: str
    message: str
    current_value: Any
    limit_value: Any
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "guard_type": self.guard_type,
            "message": self.message,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class GuardResult:
    """
    Result of guard check

    Attributes:
        passed: Whether all guards passed
        violations: List of violations
    """
    passed: bool
    violations: List[GuardViolation] = field(default_factory=list)

    @classmethod
    def success(cls) -> "GuardResult":
        """Create successful result"""
        return cls(passed=True, violations=[])

    @classmethod
    def failure(cls, violations: List[GuardViolation]) -> "GuardResult":
        """Create failed result"""
        return cls(passed=False, violations=violations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations]
        }


# ============================================================================
# EXECUTION CONTEXT
# ============================================================================

@dataclass
class ExecutionContext:
    """
    Tracks execution state for guard checks

    Attributes:
        workflow_id: Workflow being executed
        execution_id: Current execution ID
        start_time: Execution start time
        step_count: Number of steps executed
        tool_call_count: Number of tool calls made
        total_input_tokens: Total input tokens consumed
        total_output_tokens: Total output tokens consumed
        estimated_cost: Estimated cost so far (USD)
        agent_call_stack: Stack of agent calls (for recursion detection)
    """
    workflow_id: str
    execution_id: str
    start_time: datetime = field(default_factory=utc_now)
    step_count: int = 0
    tool_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost: float = 0.0
    agent_call_stack: List[str] = field(default_factory=list)

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return duration_seconds(self.start_time, utc_now())

    def record_step(self):
        """Record a step execution"""
        self.step_count += 1

    def record_tool_call(self):
        """Record a tool call"""
        self.tool_call_count += 1

    def record_llm_usage(self, input_tokens: int, output_tokens: int, model: str):
        """Record LLM token usage"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Update cost estimate
        llm_provider = get_llm_provider()
        model_metadata = llm_provider.catalog.get_model(model)
        if model_metadata:
            cost = model_metadata.estimate_cost(input_tokens, output_tokens)
            self.estimated_cost += cost

    def enter_agent(self, agent_id: str):
        """Push agent onto call stack"""
        self.agent_call_stack.append(agent_id)

    def exit_agent(self, agent_id: str):
        """Pop agent from call stack"""
        if self.agent_call_stack and self.agent_call_stack[-1] == agent_id:
            self.agent_call_stack.pop()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "elapsed_time": self.elapsed_time(),
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": self.estimated_cost,
            "agent_call_stack": self.agent_call_stack
        }


# ============================================================================
# GUARDS
# ============================================================================

class RuntimeGuard:
    """
    Enforces runtime limits during workflow execution

    Guards:
    - Cost limit: Prevent excessive LLM API costs
    - Timeout limit: Prevent long-running executions
    - Step limit: Prevent infinite loops
    - Tool call limit: Prevent tool abuse
    - Recursion limit: Prevent stack overflow

    Usage:
        guard = RuntimeGuard(
            max_cost=10.0,
            max_timeout_seconds=300,
            max_steps=100
        )

        context = ExecutionContext(workflow_id="wf_123", execution_id="exec_456")

        # Before each step
        result = guard.check_all(context)
        if not result.passed:
            raise ExecutionGuardError(result.violations)

        # Record usage
        context.record_step()
        context.record_llm_usage(1000, 500, "claude-sonnet-4-5-20250929")
    """

    def __init__(
        self,
        max_cost: Optional[float] = None,
        max_timeout_seconds: Optional[float] = None,
        max_steps: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        max_recursion_depth: Optional[int] = None
    ):
        """
        Initialize runtime guard

        Args:
            max_cost: Maximum cost in USD (default: SystemLimits.MAX_COST_PER_EXECUTION)
            max_timeout_seconds: Maximum execution time (default: SystemLimits.MAX_EXECUTION_TIMEOUT_SECONDS)
            max_steps: Maximum steps (default: SystemLimits.MAX_WORKFLOW_STEPS)
            max_tool_calls: Maximum tool calls (default: SystemLimits.MAX_TOOL_CALLS_PER_AGENT * 10)
            max_recursion_depth: Maximum agent call stack depth (default: 10)
        """
        self.max_cost = max_cost or SystemLimits.MAX_COST_PER_EXECUTION
        self.max_timeout_seconds = max_timeout_seconds or SystemLimits.MAX_EXECUTION_TIMEOUT_SECONDS
        self.max_steps = max_steps or SystemLimits.MAX_WORKFLOW_STEPS
        self.max_tool_calls = max_tool_calls or (SystemLimits.MAX_TOOL_CALLS_PER_AGENT * 10)
        self.max_recursion_depth = max_recursion_depth or 10

        logger.info(
            f"Runtime guard initialized: "
            f"max_cost=${self.max_cost:.2f}, "
            f"max_timeout={self.max_timeout_seconds}s, "
            f"max_steps={self.max_steps}, "
            f"max_tool_calls={self.max_tool_calls}, "
            f"max_recursion_depth={self.max_recursion_depth}"
        )

    def check_all(self, context: ExecutionContext) -> GuardResult:
        """
        Check all guards

        Args:
            context: Execution context

        Returns:
            GuardResult with violations if any
        """
        violations = []

        # Check each guard
        violations.extend(self.check_cost(context))
        violations.extend(self.check_timeout(context))
        violations.extend(self.check_steps(context))
        violations.extend(self.check_tool_calls(context))
        violations.extend(self.check_recursion(context))

        if violations:
            logger.warning(f"Guard violations detected: {len(violations)} violations")
            for violation in violations:
                logger.warning(f"  - {violation.guard_type}: {violation.message}")
            return GuardResult.failure(violations)

        return GuardResult.success()

    def check_cost(self, context: ExecutionContext) -> List[GuardViolation]:
        """Check cost limit"""
        if context.estimated_cost > self.max_cost:
            return [GuardViolation(
                guard_type="cost_limit",
                message=f"Estimated cost ${context.estimated_cost:.4f} exceeds limit ${self.max_cost:.2f}",
                current_value=context.estimated_cost,
                limit_value=self.max_cost
            )]
        return []

    def check_timeout(self, context: ExecutionContext) -> List[GuardViolation]:
        """Check timeout limit"""
        elapsed = context.elapsed_time()
        if elapsed > self.max_timeout_seconds:
            return [GuardViolation(
                guard_type="timeout_limit",
                message=f"Execution time {elapsed:.1f}s exceeds limit {self.max_timeout_seconds}s",
                current_value=elapsed,
                limit_value=self.max_timeout_seconds
            )]
        return []

    def check_steps(self, context: ExecutionContext) -> List[GuardViolation]:
        """Check step limit"""
        if context.step_count > self.max_steps:
            return [GuardViolation(
                guard_type="step_limit",
                message=f"Step count {context.step_count} exceeds limit {self.max_steps}",
                current_value=context.step_count,
                limit_value=self.max_steps
            )]
        return []

    def check_tool_calls(self, context: ExecutionContext) -> List[GuardViolation]:
        """Check tool call limit"""
        if context.tool_call_count > self.max_tool_calls:
            return [GuardViolation(
                guard_type="tool_call_limit",
                message=f"Tool call count {context.tool_call_count} exceeds limit {self.max_tool_calls}",
                current_value=context.tool_call_count,
                limit_value=self.max_tool_calls
            )]
        return []

    def check_recursion(self, context: ExecutionContext) -> List[GuardViolation]:
        """Check recursion depth"""
        depth = len(context.agent_call_stack)
        if depth > self.max_recursion_depth:
            return [GuardViolation(
                guard_type="recursion_limit",
                message=f"Agent call stack depth {depth} exceeds limit {self.max_recursion_depth}",
                current_value=depth,
                limit_value=self.max_recursion_depth
            )]
        return []


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_guards(
    execution_context: ExecutionContext,
    max_cost: Optional[float] = None,
    max_timeout_seconds: Optional[float] = None,
    max_steps: Optional[int] = None
) -> GuardResult:
    """
    Convenience function to check guards

    Args:
        execution_context: Execution context
        max_cost: Maximum cost override
        max_timeout_seconds: Maximum timeout override
        max_steps: Maximum steps override

    Returns:
        GuardResult
    """
    guard = RuntimeGuard(
        max_cost=max_cost,
        max_timeout_seconds=max_timeout_seconds,
        max_steps=max_steps
    )
    return guard.check_all(execution_context)


class ExecutionGuardError(Exception):
    """
    Exception raised when guards are violated

    Attributes:
        violations: List of guard violations
    """
    def __init__(self, violations: List[GuardViolation]):
        self.violations = violations
        messages = [v.message for v in violations]
        super().__init__(f"Guard violations: {'; '.join(messages)}")
