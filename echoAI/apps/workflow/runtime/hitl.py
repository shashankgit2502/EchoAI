"""
Human-in-the-Loop (HITL) manager.
Handles workflow interrupts and resumption for human review.
"""
from typing import Dict, Any, Optional


class HITLManager:
    """
    Human-in-the-Loop manager.
    Manages workflow pauses, human reviews, and resumption.
    """

    def __init__(self):
        """Initialize HITL manager."""
        self.pending_reviews: Dict[str, Dict[str, Any]] = {}

    def should_interrupt(
        self,
        workflow: Dict[str, Any],
        current_agent: str
    ) -> bool:
        """
        Check if workflow should pause for human review.

        Args:
            workflow: Workflow definition
            current_agent: Current agent being executed

        Returns:
            True if should interrupt, False otherwise
        """
        hitl_config = workflow.get("human_in_loop", {})
        if not hitl_config.get("enabled"):
            return False

        review_points = hitl_config.get("review_points", [])
        return current_agent in review_points

    def interrupt(
        self,
        run_id: str,
        workflow_id: str,
        agent_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interrupt workflow for human review.

        Args:
            run_id: Execution run identifier
            workflow_id: Workflow identifier
            agent_id: Agent at which to interrupt
            context: Current execution context

        Returns:
            Interrupt info for frontend
        """
        interrupt_data = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "context": context,
            "status": "waiting_for_review"
        }

        self.pending_reviews[run_id] = interrupt_data

        return {
            "interrupted": True,
            "run_id": run_id,
            "review_required": True,
            "agent": agent_id
        }

    def resume(
        self,
        run_id: str,
        approval: bool,
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resume workflow after human review.

        Args:
            run_id: Execution run identifier
            approval: Whether to continue or abort
            feedback: Optional human feedback

        Returns:
            Resumption result
        """
        if run_id not in self.pending_reviews:
            raise ValueError(f"No pending review for run_id: {run_id}")

        interrupt_data = self.pending_reviews.pop(run_id)

        if not approval:
            return {
                "resumed": False,
                "run_id": run_id,
                "status": "aborted_by_user"
            }

        # Resume execution with feedback
        return {
            "resumed": True,
            "run_id": run_id,
            "status": "continuing",
            "feedback": feedback
        }

    def get_pending_review(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pending review details.

        Args:
            run_id: Execution run identifier

        Returns:
            Pending review data or None
        """
        return self.pending_reviews.get(run_id)
