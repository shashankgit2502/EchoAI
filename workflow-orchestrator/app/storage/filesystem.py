"""
Filesystem Storage for Agent Systems
Handles draft/temp/final lifecycle with versioning
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from app.schemas.api_models import AgentSystemDesign, SaveWorkflowResponse
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowStorage:
    """
    Manages workflow storage with lifecycle states:
    - DRAFT: Editable, not validated
    - TEMP: Validated, for testing
    - FINAL: Immutable, versioned, production-ready
    """

    def __init__(self):
        settings = get_settings()
        self.base_path = settings.WORKFLOWS_PATH

        # Ensure directory structure exists
        self.draft_path = self.base_path / "draft"
        self.temp_path = self.base_path / "temp"
        self.final_path = self.base_path / "final"
        self.archive_path = self.base_path / "archive"

        for path in [self.draft_path, self.temp_path, self.final_path, self.archive_path]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Workflow storage initialized at: {self.base_path}")

    def save_draft(
        self,
        workflow_id: str,
        agent_system: AgentSystemDesign
    ) -> SaveWorkflowResponse:
        """
        Save as DRAFT (editable, not validated)

        Args:
            workflow_id: Unique identifier for this workflow
            agent_system: The agent system design

        Returns:
            SaveWorkflowResponse with file path
        """
        file_path = self.draft_path / f"{workflow_id}.draft.json"

        # Add metadata
        data = agent_system.model_dump()
        data["_metadata"] = {
            "state": "draft",
            "workflow_id": workflow_id,
            "saved_at": datetime.utcnow().isoformat(),
            "version": None
        }

        self._write_json(file_path, data)

        logger.info(f"Draft saved: {workflow_id}")

        return SaveWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            state="draft",
            version=None,
            file_path=str(file_path)
        )

    def save_temp(
        self,
        workflow_id: str,
        agent_system: AgentSystemDesign
    ) -> SaveWorkflowResponse:
        """
        Save as TEMP (validated, for testing)

        TEMP workflows overwrite - no versioning
        Used for iterative testing before finalizing

        Args:
            workflow_id: Unique identifier
            agent_system: The agent system design

        Returns:
            SaveWorkflowResponse
        """
        file_path = self.temp_path / f"{workflow_id}.temp.json"

        # Add metadata
        data = agent_system.model_dump()
        data["_metadata"] = {
            "state": "temp",
            "workflow_id": workflow_id,
            "saved_at": datetime.utcnow().isoformat(),
            "version": "testing"
        }

        self._write_json(file_path, data)

        logger.info(f"Temp saved: {workflow_id}")

        return SaveWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            state="temp",
            version="testing",
            file_path=str(file_path)
        )

    def save_final(
        self,
        workflow_id: str,
        agent_system: AgentSystemDesign,
        version: Optional[str] = None
    ) -> SaveWorkflowResponse:
        """
        Save as FINAL (immutable, versioned)

        If version is not provided, auto-generates next version number
        Archives old versions if they exist

        Args:
            workflow_id: Unique identifier
            agent_system: The agent system design
            version: Optional version string (e.g., "1.0", "2.1")

        Returns:
            SaveWorkflowResponse with version info
        """
        # Determine version
        if version is None:
            version = self._get_next_version(workflow_id)

        file_path = self.final_path / f"{workflow_id}.v{version}.final.json"

        # Check if this version already exists
        if file_path.exists():
            raise FileExistsError(
                f"Version {version} of workflow {workflow_id} already exists. "
                f"Cannot overwrite FINAL workflows."
            )

        # Add metadata
        data = agent_system.model_dump()
        data["_metadata"] = {
            "state": "final",
            "workflow_id": workflow_id,
            "version": version,
            "saved_at": datetime.utcnow().isoformat(),
            "immutable": True
        }

        self._write_json(file_path, data)

        logger.info(f"Final saved: {workflow_id} v{version}")

        return SaveWorkflowResponse(
            success=True,
            workflow_id=workflow_id,
            state="final",
            version=version,
            file_path=str(file_path)
        )

    def load_draft(self, workflow_id: str) -> AgentSystemDesign:
        """Load DRAFT workflow"""
        file_path = self.draft_path / f"{workflow_id}.draft.json"
        return self._load_agent_system(file_path)

    def load_temp(self, workflow_id: str) -> AgentSystemDesign:
        """Load TEMP workflow"""
        file_path = self.temp_path / f"{workflow_id}.temp.json"
        return self._load_agent_system(file_path)

    def load_final(
        self,
        workflow_id: str,
        version: Optional[str] = None
    ) -> AgentSystemDesign:
        """
        Load FINAL workflow

        If version is None, loads the latest version

        Args:
            workflow_id: Workflow identifier
            version: Optional version string

        Returns:
            AgentSystemDesign
        """
        if version is None:
            version = self._get_latest_version(workflow_id)
            if version is None:
                raise FileNotFoundError(f"No FINAL versions found for {workflow_id}")

        file_path = self.final_path / f"{workflow_id}.v{version}.final.json"
        return self._load_agent_system(file_path)

    def clone_final_to_draft(
        self,
        workflow_id: str,
        from_version: str
    ) -> AgentSystemDesign:
        """
        Clone a FINAL workflow to DRAFT for editing

        This is the ONLY way to edit FINAL workflows

        Args:
            workflow_id: Workflow identifier
            from_version: Version to clone from

        Returns:
            Cloned AgentSystemDesign (now in DRAFT state)
        """
        # Load the FINAL version
        final_system = self.load_final(workflow_id, from_version)

        # Save as DRAFT
        self.save_draft(workflow_id, final_system)

        logger.info(f"Cloned {workflow_id} v{from_version} to DRAFT")

        return final_system

    def list_versions(self, workflow_id: str) -> List[str]:
        """
        List all FINAL versions of a workflow

        Returns:
            List of version strings (e.g., ["1.0", "1.1", "2.0"])
        """
        pattern = f"{workflow_id}.v*.final.json"
        files = list(self.final_path.glob(pattern))

        versions = []
        for file in files:
            # Extract version from filename
            # Format: {workflow_id}.v{version}.final.json
            name = file.stem  # Remove .json
            parts = name.split(".v")
            if len(parts) == 2:
                version = parts[1].replace(".final", "")
                versions.append(version)

        # Sort versions (simple string sort, may need semantic versioning later)
        versions.sort()

        return versions

    def delete_draft(self, workflow_id: str) -> bool:
        """Delete DRAFT workflow"""
        file_path = self.draft_path / f"{workflow_id}.draft.json"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted DRAFT: {workflow_id}")
            return True
        return False

    def delete_temp(self, workflow_id: str) -> bool:
        """Delete TEMP workflow"""
        file_path = self.temp_path / f"{workflow_id}.temp.json"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted TEMP: {workflow_id}")
            return True
        return False

    def archive_version(self, workflow_id: str, version: str) -> bool:
        """
        Archive a FINAL version (move to archive/)

        Note: Cannot delete FINAL versions, only archive them

        Args:
            workflow_id: Workflow identifier
            version: Version to archive

        Returns:
            True if archived successfully
        """
        source = self.final_path / f"{workflow_id}.v{version}.final.json"
        dest = self.archive_path / f"{workflow_id}.v{version}.final.json"

        if source.exists():
            shutil.move(str(source), str(dest))
            logger.info(f"Archived {workflow_id} v{version}")
            return True

        return False

    def _get_next_version(self, workflow_id: str) -> str:
        """
        Auto-generate next version number

        Simple versioning: 1.0, 2.0, 3.0, etc.
        """
        versions = self.list_versions(workflow_id)

        if not versions:
            return "1.0"

        # Extract major version numbers
        major_versions = []
        for v in versions:
            try:
                major = int(v.split(".")[0])
                major_versions.append(major)
            except (ValueError, IndexError):
                continue

        if major_versions:
            next_major = max(major_versions) + 1
            return f"{next_major}.0"

        return "1.0"

    def _get_latest_version(self, workflow_id: str) -> Optional[str]:
        """Get the latest version of a workflow"""
        versions = self.list_versions(workflow_id)
        return versions[-1] if versions else None

    def _write_json(self, file_path: Path, data: dict) -> None:
        """Atomic JSON write"""
        # Write to temp file first
        temp_path = file_path.with_suffix(".tmp")

        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(file_path)

    def _load_agent_system(self, file_path: Path) -> AgentSystemDesign:
        """Load agent system from JSON file"""
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Remove internal metadata before parsing
        data.pop("_metadata", None)

        return AgentSystemDesign(**data)

    def list_all_workflows(self, state: Optional[str] = None) -> List[dict]:
        """
        List all workflows with optional state filter

        Args:
            state: Optional state filter ("draft", "temp", "final")

        Returns:
            List of workflow metadata dictionaries
        """
        workflows = []

        # Helper to extract metadata from file
        def get_workflow_info(file_path: Path, expected_state: str) -> dict:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                metadata = data.get("_metadata", {})

                # Extract workflow_id from metadata or filename
                workflow_id = metadata.get("workflow_id")
                if not workflow_id:
                    # Fallback to filename parsing
                    workflow_id = file_path.stem.split(".")[0]

                return {
                    "workflow_id": workflow_id,
                    "state": metadata.get("state", expected_state),
                    "version": metadata.get("version"),
                    "saved_at": metadata.get("saved_at"),
                    "file_path": str(file_path)
                }
            except Exception as e:
                logger.warning(f"Failed to read workflow {file_path}: {e}")
                return None

        # Scan draft directory
        if state is None or state == "draft":
            for file_path in self.draft_path.glob("*.draft.json"):
                info = get_workflow_info(file_path, "draft")
                if info:
                    workflows.append(info)

        # Scan temp directory
        if state is None or state == "temp":
            for file_path in self.temp_path.glob("*.temp.json"):
                info = get_workflow_info(file_path, "temp")
                if info:
                    workflows.append(info)

        # Scan final directory
        if state is None or state == "final":
            for file_path in self.final_path.glob("*.final.json"):
                info = get_workflow_info(file_path, "final")
                if info:
                    workflows.append(info)

        # Sort by saved_at (most recent first)
        workflows.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

        return workflows


class AgentStorage:
    """
    Storage for individual agent definitions
    (Separate from workflows for reusability)
    """

    def __init__(self):
        settings = get_settings()
        self.agents_path = settings.AGENTS_PATH
        self.agents_path.mkdir(parents=True, exist_ok=True)

    def save_agent(self, agent_id: str, agent_data: dict) -> Path:
        """Save individual agent definition"""
        file_path = self.agents_path / f"{agent_id}.json"

        data = {
            **agent_data,
            "_metadata": {
                "agent_id": agent_id,
                "saved_at": datetime.utcnow().isoformat()
            }
        }

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return file_path

    def load_agent(self, agent_id: str) -> dict:
        """Load individual agent definition"""
        file_path = self.agents_path / f"{agent_id}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Agent not found: {agent_id}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        data.pop("_metadata", None)
        return data
