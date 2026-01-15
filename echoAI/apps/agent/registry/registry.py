"""
Agent registry.
Centralized storage and retrieval for agent definitions.
"""
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class AgentRegistry:
    """
    Agent registry service.
    Manages agent definitions with persistence.
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize registry.

        Args:
            storage_dir: Directory for agent storage
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "storage" / "agents"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _agent_filename(self, agent_id: str) -> str:
        """Generate agent filename."""
        return f"{agent_id}.json"

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically write JSON to file.

        Args:
            path: Target file path
            data: Data to write
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            suffix=".tmp"
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        os.replace(tmp_path, path)

    def _load_all(self) -> None:
        """Load all agents from storage into cache."""
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file) as f:
                    agent = json.load(f)
                    agent_id = agent.get("agent_id") or agent.get("id")
                    if agent_id:
                        self._cache[agent_id] = agent
            except Exception as e:
                # Skip corrupted files
                print(f"Warning: Failed to load agent from {file}: {e}")

    def register_agent(self, agent: Dict[str, Any]) -> Dict[str, str]:
        """
        Register a new agent.

        Args:
            agent: Agent definition

        Returns:
            dict with agent_id and path
        """
        agent_id = agent.get("agent_id") or agent.get("id")
        if not agent_id:
            raise ValueError("Agent must have agent_id or id")

        # Add metadata
        if "metadata" not in agent:
            agent["metadata"] = {}

        agent["metadata"]["registered_at"] = datetime.utcnow().isoformat()

        # Save to storage
        path = self.storage_dir / self._agent_filename(agent_id)
        self._atomic_write_json(path, agent)

        # Update cache
        self._cache[agent_id] = agent

        return {
            "agent_id": agent_id,
            "path": str(path)
        }

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent definition or None
        """
        return self._cache.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.

        Returns:
            List of agent definitions
        """
        return list(self._cache.values())

    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing agent.

        Args:
            agent_id: Agent identifier
            updates: Fields to update

        Returns:
            Updated agent definition

        Raises:
            ValueError: If agent not found
        """
        if agent_id not in self._cache:
            raise ValueError(f"Agent '{agent_id}' not found")

        agent = self._cache[agent_id].copy()
        agent.update(updates)

        # Update metadata
        if "metadata" not in agent:
            agent["metadata"] = {}
        agent["metadata"]["updated_at"] = datetime.utcnow().isoformat()

        # Save to storage
        path = self.storage_dir / self._agent_filename(agent_id)
        self._atomic_write_json(path, agent)

        # Update cache
        self._cache[agent_id] = agent

        return agent

    def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent.

        Args:
            agent_id: Agent identifier

        Raises:
            ValueError: If agent not found
        """
        if agent_id not in self._cache:
            raise ValueError(f"Agent '{agent_id}' not found")

        # Remove from storage
        path = self.storage_dir / self._agent_filename(agent_id)
        if path.exists():
            os.remove(path)

        # Remove from cache
        del self._cache[agent_id]

    def get_agents_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get agents by role.

        Args:
            role: Agent role

        Returns:
            List of matching agents
        """
        return [
            agent for agent in self._cache.values()
            if agent.get("role") == role
        ]

    def get_agents_for_workflow(self, agent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple agents for a workflow.

        Args:
            agent_ids: List of agent identifiers

        Returns:
            Dict mapping agent_id to agent definition
        """
        result = {}
        for agent_id in agent_ids:
            agent = self.get_agent(agent_id)
            if agent:
                result[agent_id] = agent
        return result

    def validate_agent(self, agent: Dict[str, Any]) -> bool:
        """
        Validate agent definition against schema.

        Args:
            agent: Agent definition

        Returns:
            True if valid

        TODO: Implement actual schema validation
        """
        required_fields = ["agent_id", "name", "role"]
        return all(field in agent for field in required_fields)
