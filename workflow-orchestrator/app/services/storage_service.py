"""
Storage Service
Service layer wrapper for workflow persistence
Provides clean interface for internal API communication
"""
from typing import Optional, List
from app.schemas.api_models import (
    AgentSystemDesign,
    SaveWorkflowRequest,
    SaveWorkflowResponse,
    LoadWorkflowRequest,
    ListWorkflowsResponse,
    DeleteWorkflowRequest,
    ArchiveWorkflowRequest,
    CloneWorkflowRequest
)
from app.storage.filesystem import WorkflowStorage
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    """
    Service layer for storage operations

    Enforces service boundaries:
    - No direct imports in API layer
    - DTO-based request/response
    - Idempotent
    - Ready for microservice extraction

    Responsibilities:
    - Workflow persistence (draft/temp/final)
    - Version management
    - Cloning operations
    """

    def __init__(self):
        """Initialize storage service"""
        self._storage = WorkflowStorage()
        logger.info("StorageService initialized")

    def save_draft(
        self,
        request: SaveWorkflowRequest
    ) -> SaveWorkflowResponse:
        """
        Save workflow as DRAFT

        Args:
            request: Save request

        Returns:
            SaveWorkflowResponse with save status
        """
        logger.info(f"Saving DRAFT: {request.workflow_id}")

        result = self._storage.save_draft(
            request.workflow_id,
            request.agent_system
        )

        return result

    def save_temp(
        self,
        request: SaveWorkflowRequest
    ) -> SaveWorkflowResponse:
        """
        Save workflow as TEMP

        Args:
            request: Save request

        Returns:
            SaveWorkflowResponse with save status
        """
        logger.info(f"Saving TEMP: {request.workflow_id}")

        result = self._storage.save_temp(
            request.workflow_id,
            request.agent_system
        )

        return result

    def save_final(
        self,
        request: SaveWorkflowRequest
    ) -> SaveWorkflowResponse:
        """
        Save workflow as FINAL (versioned, immutable)

        Also removes draft version if it exists (draft → final transition)

        Args:
            request: Save request with version

        Returns:
            SaveWorkflowResponse with save status
        """
        logger.info(f"Saving FINAL: {request.workflow_id} v{request.version}")

        result = self._storage.save_final(
            request.workflow_id,
            request.agent_system,
            request.version
        )

        # Auto-remove draft if it exists (draft → final transition)
        if result.success:
            draft_deleted = self._storage.delete_draft(request.workflow_id)
            if draft_deleted:
                logger.info(f"Auto-deleted draft after saving as final: {request.workflow_id}")

            # Also remove temp if it exists
            temp_deleted = self._storage.delete_temp(request.workflow_id)
            if temp_deleted:
                logger.info(f"Auto-deleted temp after saving as final: {request.workflow_id}")

        return result

    def load_workflow(
        self,
        request: LoadWorkflowRequest
    ) -> Optional[AgentSystemDesign]:
        """
        Load workflow from storage

        Args:
            request: Load request

        Returns:
            AgentSystemDesign or None
        """
        logger.info(f"Loading workflow: {request.workflow_id} ({request.state})")

        if request.state == "draft":
            return self._storage.load_draft(request.workflow_id)
        elif request.state == "temp":
            return self._storage.load_temp(request.workflow_id)
        elif request.state == "final":
            return self._storage.load_final(request.workflow_id, request.version)
        else:
            return None

    def list_workflows(
        self,
        state: Optional[str] = None
    ) -> ListWorkflowsResponse:
        """
        List all workflows

        Args:
            state: Optional state filter (draft/temp/final)

        Returns:
            ListWorkflowsResponse with workflows
        """
        logger.info(f"Listing workflows (state: {state})")

        workflows = self._storage.list_all_workflows(state)

        return ListWorkflowsResponse(
            workflows=workflows,
            count=len(workflows)
        )

    def list_versions(
        self,
        workflow_id: str
    ) -> List[str]:
        """
        List all versions of a workflow

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of version strings
        """
        logger.info(f"Listing versions: {workflow_id}")
        return self._storage.list_versions(workflow_id)

    def clone_final_to_draft(
        self,
        request: CloneWorkflowRequest
    ) -> AgentSystemDesign:
        """
        Clone FINAL workflow to DRAFT for editing

        Args:
            request: Clone request

        Returns:
            Cloned AgentSystemDesign
        """
        logger.info(f"Cloning FINAL→DRAFT: {request.workflow_id} v{request.from_version}")

        result = self._storage.clone_final_to_draft(
            request.workflow_id,
            request.from_version
        )

        return result

    def delete_workflow(
        self,
        request: DeleteWorkflowRequest
    ) -> bool:
        """
        Delete workflow (draft/temp only)

        Args:
            request: Delete request

        Returns:
            True if deleted, False otherwise
        """
        logger.info(f"Deleting workflow: {request.workflow_id} ({request.state})")

        if request.state == "draft":
            return self._storage.delete_draft(request.workflow_id)
        elif request.state == "temp":
            return self._storage.delete_temp(request.workflow_id)
        else:
            return False

    def archive_version(
        self,
        request: ArchiveWorkflowRequest
    ) -> bool:
        """
        Archive FINAL version

        Args:
            request: Archive request

        Returns:
            True if archived, False otherwise
        """
        logger.info(f"Archiving version: {request.workflow_id} v{request.version}")

        return self._storage.archive_version(
            request.workflow_id,
            request.version
        )


# ============================================================================
# SINGLETON INSTANCE (optional, or use dependency injection)
# ============================================================================

_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """
    Get singleton storage service instance

    Returns:
        StorageService instance
    """
    global _storage_service

    if _storage_service is None:
        _storage_service = StorageService()

    return _storage_service
