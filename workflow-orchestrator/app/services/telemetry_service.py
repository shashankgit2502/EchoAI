"""
Telemetry Service
Service layer wrapper for execution metrics and observability
Provides clean interface for internal API communication
"""
from typing import Optional, List
from datetime import datetime
from app.schemas.api_models import (
    TelemetryQuery,
    TelemetryResponse,
    ExecutionMetrics,
    AgentMetrics
)
from app.runtime.telemetry import RuntimeTelemetry
from app.core.logging import get_logger

logger = get_logger(__name__)


class TelemetryService:
    """
    Service layer for telemetry operations

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Async + idempotent
    - Ready for microservice extraction

    Responsibilities:
    - Metrics collection
    - Execution tracing
    - Performance analysis
    - Cost tracking (INR)
    """

    def __init__(self):
        """Initialize telemetry service"""
        self._collector = RuntimeTelemetry()
        logger.info("TelemetryService initialized")

    async def query_metrics(
        self,
        query: TelemetryQuery
    ) -> TelemetryResponse:
        """
        Query execution metrics

        Args:
            query: Telemetry query with filters

        Returns:
            TelemetryResponse with metrics
        """
        logger.info(f"Querying metrics (workflow: {query.workflow_id}, run: {query.run_id})")

        # Collect execution metrics
        execution_metrics = None
        agent_metrics = []
        spans = []

        if query.run_id:
            # Get metrics for specific run
            run_metrics = await self._collector.get_run_metrics(query.run_id)
            if run_metrics:
                execution_metrics = ExecutionMetrics(
                    total_executions=1,
                    successful_executions=1 if run_metrics.get("status") == "completed" else 0,
                    failed_executions=1 if run_metrics.get("status") == "failed" else 0,
                    avg_duration_seconds=run_metrics.get("duration_seconds", 0),
                    total_cost_inr=run_metrics.get("total_cost_inr", 0),
                    total_tokens=run_metrics.get("total_tokens", 0)
                )

                # Get agent-specific metrics
                for agent_id, metrics in run_metrics.get("agent_metrics", {}).items():
                    agent_metrics.append(AgentMetrics(
                        agent_id=agent_id,
                        invocations=metrics.get("invocations", 0),
                        avg_duration_seconds=metrics.get("avg_duration", 0),
                        total_cost_inr=metrics.get("cost_inr", 0),
                        success_rate=metrics.get("success_rate", 1.0)
                    ))

        elif query.workflow_id:
            # Get aggregated metrics for workflow
            workflow_metrics = await self._collector.get_workflow_metrics(
                query.workflow_id,
                query.start_time,
                query.end_time
            )
            if workflow_metrics:
                execution_metrics = ExecutionMetrics(**workflow_metrics)

        logger.info(f"Metrics retrieved: {len(agent_metrics)} agents")

        return TelemetryResponse(
            execution_metrics=execution_metrics,
            agent_metrics=agent_metrics,
            spans=spans
        )

    async def get_workflow_history(
        self,
        workflow_id: str,
        limit: int = 100
    ) -> List[dict]:
        """
        Get execution history for workflow

        Args:
            workflow_id: Workflow identifier
            limit: Maximum number of results

        Returns:
            List of execution records
        """
        logger.info(f"Getting execution history: {workflow_id} (limit: {limit})")

        history = await self._collector.get_execution_history(workflow_id, limit)
        return history

    async def get_cost_breakdown(
        self,
        run_id: str
    ) -> dict:
        """
        Get cost breakdown for execution

        Args:
            run_id: Execution run ID

        Returns:
            Cost breakdown by agent/tool (in INR)
        """
        logger.info(f"Getting cost breakdown: {run_id}")

        breakdown = await self._collector.get_cost_breakdown(run_id)
        return breakdown


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_telemetry_service: Optional[TelemetryService] = None


def get_telemetry_service() -> TelemetryService:
    """
    Get singleton telemetry service instance

    Returns:
        TelemetryService instance
    """
    global _telemetry_service

    if _telemetry_service is None:
        _telemetry_service = TelemetryService()

    return _telemetry_service
