"""
Agent Registry
Load, cache, and manage agent definitions from JSON files
"""
import json
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.constants import AgentRole
from app.schemas.api_models import AgentDefinition

logger = get_logger(__name__)


# ============================================================================
# AGENT METADATA
# ============================================================================

@dataclass
class AgentMetadata:
    """
    Extended agent metadata with runtime information
    """
    agent_id: str
    definition: AgentDefinition
    file_path: Path
    loaded_at: str
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "definition": self.definition.dict(),
            "file_path": str(self.file_path),
            "loaded_at": self.loaded_at,
            "version": self.version
        }


# ============================================================================
# AGENT REGISTRY
# ============================================================================

class AgentRegistry:
    """
    Registry for loading and caching agent definitions

    Loads agent JSON files from storage and provides lookup methods.
    Caches loaded agents for performance.

    Usage:
        registry = AgentRegistry()

        # Load single agent
        agent = registry.load_agent("security_analyst")

        # Load all agents
        agents = registry.load_all_agents()

        # Get agent by ID
        agent = registry.get_agent("security_analyst")

        # List available agents
        agent_ids = registry.list_agents()
    """

    def __init__(self, agents_path: Optional[Path] = None):
        """
        Initialize agent registry

        Args:
            agents_path: Path to agents directory (defaults to config.AGENTS_PATH)
        """
        self.settings = get_settings()
        self.agents_path = agents_path or self.settings.AGENTS_PATH
        self._cache: Dict[str, AgentMetadata] = {}
        self._loaded = False

        logger.info(f"Agent registry initialized: {self.agents_path}")

    def _load_agent_from_file(self, file_path: Path) -> Optional[AgentMetadata]:
        """
        Load agent definition from JSON file

        Args:
            file_path: Path to agent JSON file

        Returns:
            AgentMetadata or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Parse agent definition
            agent_def = AgentDefinition(**data)

            # Create metadata
            metadata = AgentMetadata(
                agent_id=agent_def.id,
                definition=agent_def,
                file_path=file_path,
                loaded_at=str(file_path.stat().st_mtime),
                version=data.get("version", "1.0")
            )

            logger.debug(f"Loaded agent: {agent_def.id} from {file_path.name}")

            return metadata

        except Exception as e:
            logger.error(f"Failed to load agent from {file_path}: {e}", exc_info=True)
            return None

    def load_agent(self, agent_id: str) -> Optional[AgentDefinition]:
        """
        Load specific agent by ID

        Args:
            agent_id: Agent ID

        Returns:
            AgentDefinition or None if not found
        """
        # Check cache
        if agent_id in self._cache:
            return self._cache[agent_id].definition

        # Find and load agent file
        agent_file = self.agents_path / f"{agent_id}.json"

        if not agent_file.exists():
            logger.warning(f"Agent file not found: {agent_file}")
            return None

        # Load agent
        metadata = self._load_agent_from_file(agent_file)
        if metadata:
            self._cache[agent_id] = metadata
            return metadata.definition

        return None

    def load_all_agents(self, force_reload: bool = False) -> List[AgentDefinition]:
        """
        Load all agents from agents directory

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            List of agent definitions
        """
        if self._loaded and not force_reload:
            return [meta.definition for meta in self._cache.values()]

        # Clear cache if reloading
        if force_reload:
            self._cache.clear()

        # Ensure agents directory exists
        if not self.agents_path.exists():
            logger.warning(f"Agents directory not found: {self.agents_path}")
            self.agents_path.mkdir(parents=True, exist_ok=True)
            return []

        # Load all agent JSON files
        agent_files = list(self.agents_path.glob("*.json"))

        logger.info(f"Loading {len(agent_files)} agents from {self.agents_path}")

        loaded_count = 0
        for agent_file in agent_files:
            metadata = self._load_agent_from_file(agent_file)
            if metadata:
                self._cache[metadata.agent_id] = metadata
                loaded_count += 1

        self._loaded = True

        logger.info(f"Loaded {loaded_count}/{len(agent_files)} agents successfully")

        return [meta.definition for meta in self._cache.values()]

    def get_agent(self, agent_id: str) -> Optional[AgentDefinition]:
        """
        Get agent from cache (load if not cached)

        Args:
            agent_id: Agent ID

        Returns:
            AgentDefinition or None if not found
        """
        # Check cache
        if agent_id in self._cache:
            return self._cache[agent_id].definition

        # Try to load
        return self.load_agent(agent_id)

    def list_agents(self) -> List[str]:
        """
        List all available agent IDs

        Returns:
            List of agent IDs
        """
        # Ensure agents are loaded
        if not self._loaded:
            self.load_all_agents()

        return list(self._cache.keys())

    def get_agents_by_role(self, role: AgentRole) -> List[AgentDefinition]:
        """
        Get all agents with a specific role

        Args:
            role: Agent role

        Returns:
            List of agent definitions
        """
        # Ensure agents are loaded
        if not self._loaded:
            self.load_all_agents()

        return [
            meta.definition
            for meta in self._cache.values()
            if meta.definition.role == role.value
        ]

    def get_master_agents(self) -> List[AgentDefinition]:
        """
        Get all master agents (coordinators)

        Returns:
            List of master agent definitions
        """
        # Ensure agents are loaded
        if not self._loaded:
            self.load_all_agents()

        return [
            meta.definition
            for meta in self._cache.values()
            if meta.definition.is_master
        ]

    def save_agent(self, agent: AgentDefinition) -> Path:
        """
        Save agent definition to file

        Args:
            agent: Agent definition

        Returns:
            Path to saved file
        """
        # Ensure agents directory exists
        self.agents_path.mkdir(parents=True, exist_ok=True)

        # Build file path
        file_path = self.agents_path / f"{agent.id}.json"

        # Save to file
        try:
            with open(file_path, 'w') as f:
                json.dump(agent.dict(), f, indent=2)

            logger.info(f"Saved agent: {agent.id} to {file_path}")

            # Update cache
            metadata = AgentMetadata(
                agent_id=agent.id,
                definition=agent,
                file_path=file_path,
                loaded_at=str(file_path.stat().st_mtime),
                version="1.0"
            )
            self._cache[agent.id] = metadata

            return file_path

        except Exception as e:
            logger.error(f"Failed to save agent {agent.id}: {e}", exc_info=True)
            raise

    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete agent definition

        Args:
            agent_id: Agent ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.agents_path / f"{agent_id}.json"

        if not file_path.exists():
            logger.warning(f"Agent file not found for deletion: {file_path}")
            return False

        try:
            file_path.unlink()
            logger.info(f"Deleted agent: {agent_id}")

            # Remove from cache
            if agent_id in self._cache:
                del self._cache[agent_id]

            return True

        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}", exc_info=True)
            return False

    def reload(self):
        """Reload all agents from disk"""
        logger.info("Reloading all agents...")
        self._loaded = False
        self.load_all_agents(force_reload=True)


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

_global_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """
    Get global agent registry instance

    Singleton pattern for reusing loaded agents

    Returns:
        AgentRegistry
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = AgentRegistry()

    return _global_registry
