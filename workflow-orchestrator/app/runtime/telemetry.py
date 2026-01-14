"""
Runtime Telemetry
Tracks workflow execution metrics for observability
Emits spans for workflow/agent/tool runs
Phase 9 implementation - uses basic logging (Phase 10 will add full OTEL)

Note: All costs are tracked in Indian Rupees (INR)
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from app.core.logging import get_logger
from app.utils.time import utc_now, duration_ms, duration_seconds
from app.utils.ids import generate_id

logger = get_logger(__name__)


# ============================================================================
# TELEMETRY DATA MODELS
# ============================================================================

@dataclass
class ExecutionSpan:
    """
    Execution span for telemetry

    Attributes:
        span_id: Unique span ID
        span_type: Type (workflow, agent, tool)
        name: Span name
        run_id: Execution run ID
        parent_span_id: Parent span ID (for nested spans)
        start_time: Start timestamp
        end_time: End timestamp
        duration_ms: Duration in milliseconds
        status: Status (success, error, timeout)
        attributes: Additional attributes
        error: Error message if failed
    """
    span_id: str
    span_type: str
    name: str
    run_id: str
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=utc_now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, status: str = "success", error: Optional[str] = None):
        """Mark span as finished"""
        self.end_time = utc_now()
        self.duration_ms = duration_ms(self.start_time, self.end_time)
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "span_id": self.span_id,
            "span_type": self.span_type,
            "name": self.name,
            "run_id": self.run_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "error": self.error
        }


@dataclass
class ExecutionMetrics:
    """
    Aggregated execution metrics

    Attributes:
        run_id: Execution run ID
        workflow_id: Workflow ID
        total_duration_ms: Total execution time
        agent_count: Number of agents executed
        tool_call_count: Number of tool calls
        total_input_tokens: Total input tokens
        total_output_tokens: Total output tokens
        total_cost_inr: Total cost in Indian Rupees (INR)
        error_count: Number of errors
        success: Whether execution succeeded
    """
    run_id: str
    workflow_id: str
    total_duration_ms: float = 0.0
    agent_count: int = 0
    tool_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_inr: float = 0.0
    error_count: int = 0
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "total_duration_ms": self.total_duration_ms,
            "agent_count": self.agent_count,
            "tool_call_count": self.tool_call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_inr": self.total_cost_inr,
            "error_count": self.error_count,
            "success": self.success
        }


# ============================================================================
# RUNTIME TELEMETRY TRACKER
# ============================================================================

class RuntimeTelemetry:
    """
    Tracks runtime execution metrics

    Phase 9: Basic implementation with logging
    Phase 10: Will add full OpenTelemetry instrumentation

    Responsibilities:
    - Create and track execution spans
    - Aggregate metrics
    - Emit logs for observability
    - Store execution history

    Usage:
        telemetry = RuntimeTelemetry()

        # Track workflow execution
        with telemetry.track_workflow(run_id="exec_123", workflow_id="wf_456") as span:
            # Execute workflow
            execute_workflow()

        # Track agent execution
        with telemetry.track_agent(run_id="exec_123", agent_id="analyst") as span:
            # Execute agent
            result = agent.invoke(input)

            # Add attributes (cost in INR)
            span.attributes["input_tokens"] = 1000
            span.attributes["output_tokens"] = 500
            span.attributes["cost_inr"] = 4.15

        # Get metrics
        metrics = telemetry.get_metrics(run_id="exec_123")
    """

    def __init__(self):
        """Initialize runtime telemetry"""
        # In-memory storage (in production, use database or OTEL backend)
        self._spans: Dict[str, List[ExecutionSpan]] = {}
        self._metrics: Dict[str, ExecutionMetrics] = {}
        self._active_spans: Dict[str, ExecutionSpan] = {}

        logger.info("Runtime telemetry initialized (Phase 9 - basic, costs in INR)")

    # ========================================================================
    # SPAN TRACKING
    # ========================================================================

    @contextmanager
    def track_workflow(
        self,
        run_id: str,
        workflow_id: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Track workflow execution

        Args:
            run_id: Execution run ID
            workflow_id: Workflow ID
            attributes: Additional attributes

        Yields:
            ExecutionSpan

        Usage:
            with telemetry.track_workflow(run_id="exec_123", workflow_id="wf_456") as span:
                execute_workflow()
                span.attributes["step_count"] = 5
        """
        span = self._start_span(
            span_type="workflow",
            name=f"workflow:{workflow_id}",
            run_id=run_id,
            attributes=attributes or {}
        )

        # Initialize metrics for this run
        if run_id not in self._metrics:
            self._metrics[run_id] = ExecutionMetrics(
                run_id=run_id,
                workflow_id=workflow_id
            )

        try:
            yield span
            span.finish(status="success")
            logger.info(
                f"Workflow execution complete: {workflow_id} "
                f"(duration: {span.duration_ms:.2f}ms)"
            )

        except Exception as e:
            span.finish(status="error", error=str(e))
            self._metrics[run_id].error_count += 1
            self._metrics[run_id].success = False
            logger.error(
                f"Workflow execution failed: {workflow_id} - {e}",
                exc_info=True
            )
            raise

        finally:
            self._finish_span(span)

            # Update metrics
            metrics = self._metrics[run_id]
            metrics.total_duration_ms = span.duration_ms or 0.0

    @contextmanager
    def track_agent(
        self,
        run_id: str,
        agent_id: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Track agent execution

        Args:
            run_id: Execution run ID
            agent_id: Agent ID
            parent_span_id: Parent span ID
            attributes: Additional attributes

        Yields:
            ExecutionSpan

        Usage:
            with telemetry.track_agent(run_id="exec_123", agent_id="analyst") as span:
                result = agent.invoke(input)
                span.attributes["input_tokens"] = result.input_tokens
                span.attributes["output_tokens"] = result.output_tokens
                span.attributes["cost_inr"] = result.cost_inr  # Cost in INR
        """
        span = self._start_span(
            span_type="agent",
            name=f"agent:{agent_id}",
            run_id=run_id,
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )

        try:
            yield span
            span.finish(status="success")
            logger.info(
                f"Agent execution complete: {agent_id} "
                f"(duration: {span.duration_ms:.2f}ms)"
            )

        except Exception as e:
            span.finish(status="error", error=str(e))
            if run_id in self._metrics:
                self._metrics[run_id].error_count += 1
            logger.error(
                f"Agent execution failed: {agent_id} - {e}",
                exc_info=True
            )
            raise

        finally:
            self._finish_span(span)

            # Update metrics
            if run_id in self._metrics:
                metrics = self._metrics[run_id]
                metrics.agent_count += 1

                # Extract token usage and cost from attributes
                if "input_tokens" in span.attributes:
                    metrics.total_input_tokens += span.attributes["input_tokens"]
                if "output_tokens" in span.attributes:
                    metrics.total_output_tokens += span.attributes["output_tokens"]
                if "cost_inr" in span.attributes:
                    metrics.total_cost_inr += span.attributes["cost_inr"]

    @contextmanager
    def track_tool(
        self,
        run_id: str,
        tool_name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Track tool execution

        Args:
            run_id: Execution run ID
            tool_name: Tool name
            parent_span_id: Parent span ID
            attributes: Additional attributes

        Yields:
            ExecutionSpan

        Usage:
            with telemetry.track_tool(run_id="exec_123", tool_name="search_db") as span:
                result = tool.invoke(params)
                span.attributes["result_count"] = len(result)
        """
        span = self._start_span(
            span_type="tool",
            name=f"tool:{tool_name}",
            run_id=run_id,
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )

        try:
            yield span
            span.finish(status="success")
            logger.debug(
                f"Tool execution complete: {tool_name} "
                f"(duration: {span.duration_ms:.2f}ms)"
            )

        except Exception as e:
            span.finish(status="error", error=str(e))
            if run_id in self._metrics:
                self._metrics[run_id].error_count += 1
            logger.error(
                f"Tool execution failed: {tool_name} - {e}",
                exc_info=True
            )
            raise

        finally:
            self._finish_span(span)

            # Update metrics
            if run_id in self._metrics:
                self._metrics[run_id].tool_call_count += 1

    # ========================================================================
    # METRICS RETRIEVAL
    # ========================================================================

    def get_metrics(self, run_id: str) -> Optional[ExecutionMetrics]:
        """Get aggregated metrics for run"""
        return self._metrics.get(run_id)

    def get_spans(self, run_id: str) -> List[ExecutionSpan]:
        """Get all spans for run"""
        return self._spans.get(run_id, [])

    def get_span_tree(self, run_id: str) -> Dict[str, Any]:
        """
        Get span tree for run (hierarchical view)

        Returns:
            Tree structure of spans
        """
        spans = self.get_spans(run_id)
        if not spans:
            return {}

        # Build tree
        tree = {}
        span_map = {s.span_id: s for s in spans}

        for span in spans:
            if span.parent_span_id is None:
                # Root span
                tree[span.span_id] = self._build_span_node(span, span_map)

        return tree

    def _build_span_node(
        self,
        span: ExecutionSpan,
        span_map: Dict[str, ExecutionSpan]
    ) -> Dict[str, Any]:
        """Build span tree node"""
        node = span.to_dict()
        node["children"] = []

        # Find children
        for other_span in span_map.values():
            if other_span.parent_span_id == span.span_id:
                child_node = self._build_span_node(other_span, span_map)
                node["children"].append(child_node)

        return node

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _start_span(
        self,
        span_type: str,
        name: str,
        run_id: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> ExecutionSpan:
        """Start new span"""
        span_id = generate_id("span")

        span = ExecutionSpan(
            span_id=span_id,
            span_type=span_type,
            name=name,
            run_id=run_id,
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )

        # Store span
        if run_id not in self._spans:
            self._spans[run_id] = []
        self._spans[run_id].append(span)

        self._active_spans[span_id] = span

        logger.debug(f"Started span: {span_type}:{name} (id={span_id})")

        return span

    def _finish_span(self, span: ExecutionSpan):
        """Finish span"""
        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]

        logger.debug(
            f"Finished span: {span.span_type}:{span.name} "
            f"(status={span.status}, duration={span.duration_ms:.2f}ms)"
        )

    def clear_run_data(self, run_id: str):
        """Clear all data for a run"""
        if run_id in self._spans:
            del self._spans[run_id]
        if run_id in self._metrics:
            del self._metrics[run_id]

        logger.info(f"Cleared telemetry data for run {run_id}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_runtime_telemetry: Optional[RuntimeTelemetry] = None


def get_runtime_telemetry() -> RuntimeTelemetry:
    """Get singleton runtime telemetry instance"""
    global _runtime_telemetry
    if _runtime_telemetry is None:
        _runtime_telemetry = RuntimeTelemetry()
    return _runtime_telemetry


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def track_execution(run_id: str, workflow_id: str):
    """
    Convenience function to track workflow execution

    Usage:
        with track_execution(run_id="exec_123", workflow_id="wf_456"):
            execute_workflow()
    """
    telemetry = get_runtime_telemetry()
    return telemetry.track_workflow(run_id=run_id, workflow_id=workflow_id)


def record_agent_execution(
    run_id: str,
    agent_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_inr: float
):
    """
    Record completed agent execution

    Args:
        run_id: Execution run ID
        agent_id: Agent ID
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        cost_inr: Cost in Indian Rupees (INR)
    """
    telemetry = get_runtime_telemetry()

    if run_id in telemetry._metrics:
        metrics = telemetry._metrics[run_id]
        metrics.agent_count += 1
        metrics.total_input_tokens += input_tokens
        metrics.total_output_tokens += output_tokens
        metrics.total_cost_inr += cost_inr

        logger.info(
            f"Recorded agent execution: {agent_id} "
            f"(tokens: {input_tokens}/{output_tokens}, cost: ₹{cost_inr:.2f})"
        )


def record_tool_execution(run_id: str, tool_name: str):
    """
    Record completed tool execution

    Args:
        run_id: Execution run ID
        tool_name: Tool name
    """
    telemetry = get_runtime_telemetry()

    if run_id in telemetry._metrics:
        metrics = telemetry._metrics[run_id]
        metrics.tool_call_count += 1

        logger.debug(f"Recorded tool execution: {tool_name}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example: Tracking workflow execution with telemetry (costs in INR)

from app.runtime.telemetry import get_runtime_telemetry

telemetry = get_runtime_telemetry()

# Track entire workflow
with telemetry.track_workflow(run_id="exec_123", workflow_id="wf_456") as workflow_span:
    # Track agent execution
    with telemetry.track_agent(
        run_id="exec_123",
        agent_id="analyst",
        parent_span_id=workflow_span.span_id
    ) as agent_span:
        # Execute agent
        result = agent.invoke(input_data)

        # Record metrics (cost in INR)
        agent_span.attributes["input_tokens"] = result.input_tokens
        agent_span.attributes["output_tokens"] = result.output_tokens
        agent_span.attributes["cost_inr"] = result.cost_inr  # Cost in Indian Rupees

        # Track tool calls
        with telemetry.track_tool(
            run_id="exec_123",
            tool_name="search_db",
            parent_span_id=agent_span.span_id
        ) as tool_span:
            tool_result = search_db(query)
            tool_span.attributes["result_count"] = len(tool_result)

# Get metrics
metrics = telemetry.get_metrics("exec_123")
print(f"Total cost: ₹{metrics.total_cost_inr:.2f}")  # Cost in INR
print(f"Total duration: {metrics.total_duration_ms:.2f}ms")
print(f"Agent count: {metrics.agent_count}")
print(f"Tool calls: {metrics.tool_call_count}")

# Get span tree
span_tree = telemetry.get_span_tree("exec_123")
"""
