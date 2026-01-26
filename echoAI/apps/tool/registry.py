"""
Tool Registry Module

This module provides the ToolRegistry class for managing tool registration,
discovery, and retrieval in the EchoAI system. It handles both local tools
(Python functions) and external tools (MCP servers, API endpoints).

The registry maintains an index of all available tools and provides methods
for querying, filtering, and validating tools for agent use.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from echolib.types import ToolDef, ToolType

from .storage import ToolStorage

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for managing tools across the EchoAI platform.

    The ToolRegistry is responsible for:
    - Registering new tools (local, MCP, API)
    - Discovering available tools from configured locations
    - Providing tool metadata and schemas
    - Filtering tools by type, capability, or agent requirements
    - Validating tool inputs against schemas

    Attributes:
        storage (ToolStorage): Persistence layer for tool definitions
        discovery_dirs (List[Path]): Directories to scan for local tools
        _cache (Dict[str, ToolDef]): In-memory cache of registered tools
    """

    def __init__(
        self,
        storage: ToolStorage,
        discovery_dirs: Optional[List[Path]] = None
    ):
        """
        Initialize the ToolRegistry.

        Args:
            storage: ToolStorage instance for persistence
            discovery_dirs: Optional list of directories to scan for tool manifests
        """
        self.storage = storage
        self.discovery_dirs = discovery_dirs or []
        self._cache: Dict[str, ToolDef] = {}

        # Load all existing tools into cache
        self._load_all()

        logger.info(
            f"ToolRegistry initialized with {len(self._cache)} tools, "
            f"{len(self.discovery_dirs)} discovery directories"
        )

    def register(self, tool: ToolDef) -> Dict[str, str]:
        """
        Register a new tool or update an existing one.

        Args:
            tool: ToolDef instance to register

        Returns:
            Dict with registration result (tool_id, status, message)

        Raises:
            ValueError: If tool definition is invalid
        """
        # Validate required fields
        if not tool.name:
            raise ValueError("Tool must have a name")

        if not tool.description:
            raise ValueError("Tool must have a description")

        # Ensure tool_id is set
        if not tool.tool_id:
            tool_id = f"tool_{tool.name.lower().replace(' ', '_').replace('-', '_')}"
            # Create a new tool with the generated ID
            tool = ToolDef(
                tool_id=tool_id,
                name=tool.name,
                description=tool.description,
                tool_type=tool.tool_type,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                execution_config=tool.execution_config,
                version=tool.version,
                tags=tool.tags,
                status=tool.status,
                metadata=tool.metadata
            )

        # Check if updating existing tool
        is_update = tool.tool_id in self._cache

        try:
            # Persist to storage
            self.storage.save_tool(tool)

            # Update cache
            self._cache[tool.tool_id] = tool

            status = "updated" if is_update else "registered"
            logger.info(f"Tool '{tool.tool_id}' {status}")

            return {
                "tool_id": tool.tool_id,
                "status": status,
                "message": f"Tool '{tool.name}' {status} successfully"
            }

        except Exception as e:
            logger.error(f"Failed to register tool '{tool.name}': {e}")
            raise ValueError(f"Failed to register tool: {e}")

    def get(self, tool_id: str) -> Optional[ToolDef]:
        """
        Retrieve a tool definition by its ID.

        Args:
            tool_id: Unique identifier of the tool

        Returns:
            ToolDef instance if found, None otherwise
        """
        # Try cache first
        if tool_id in self._cache:
            return self._cache[tool_id]

        # Try loading from storage (in case cache is stale)
        tool = self.storage.load_tool(tool_id)
        if tool:
            self._cache[tool_id] = tool
            return tool

        logger.debug(f"Tool '{tool_id}' not found")
        return None

    def get_by_name(self, name: str) -> Optional[ToolDef]:
        """
        Retrieve a tool definition by its name.

        Args:
            name: Human-readable name of the tool

        Returns:
            ToolDef instance if found, None otherwise
        """
        name_lower = name.lower()
        for tool in self._cache.values():
            if tool.name.lower() == name_lower:
                return tool
        return None

    def get_tool_id_by_name(self, name: str) -> Optional[str]:
        """
        Get tool_id by tool name.

        This method supports frontend tool resolution where tools
        are referenced by name.

        Args:
            name: Human-readable name of the tool

        Returns:
            tool_id if found, None otherwise
        """
        tool = self.get_by_name(name)
        return tool.tool_id if tool else None

    def list_all(self) -> List[ToolDef]:
        """
        List all registered tools.

        Returns:
            List of all ToolDef instances in the registry
        """
        return list(self._cache.values())

    def list_by_type(self, tool_type: ToolType) -> List[ToolDef]:
        """
        List all tools of a specific type.

        Args:
            tool_type: ToolType enum value to filter by

        Returns:
            List of ToolDef instances matching the specified type
        """
        return [
            tool for tool in self._cache.values()
            if tool.tool_type == tool_type
        ]

    def list_by_status(self, status: str) -> List[ToolDef]:
        """
        List all tools with a specific status.

        Args:
            status: Status string to filter by (active, deprecated, disabled)

        Returns:
            List of ToolDef instances with the specified status
        """
        return [
            tool for tool in self._cache.values()
            if tool.status == status.lower()
        ]

    def list_by_tags(self, tags: List[str]) -> List[ToolDef]:
        """
        List tools that have any of the specified tags.

        Args:
            tags: List of tags to filter by

        Returns:
            List of ToolDef instances that have at least one matching tag
        """
        tags_lower = {tag.lower() for tag in tags}
        return [
            tool for tool in self._cache.values()
            if any(tag.lower() in tags_lower for tag in tool.tags)
        ]

    def delete(self, tool_id: str) -> bool:
        """
        Delete a tool from the registry.

        Args:
            tool_id: Unique identifier of the tool to delete

        Returns:
            True if deletion successful, False if tool not found
        """
        if tool_id not in self._cache:
            logger.warning(f"Tool '{tool_id}' not found for deletion")
            return False

        try:
            # Delete from storage
            success = self.storage.delete_tool(tool_id)

            if success:
                # Remove from cache
                del self._cache[tool_id]
                logger.info(f"Tool '{tool_id}' deleted from registry")

            return success

        except Exception as e:
            logger.error(f"Failed to delete tool '{tool_id}': {e}")
            return False

    def discover_local_tools(self) -> List[ToolDef]:
        """
        Discover and register tools from configured discovery directories.

        Scans each directory in discovery_dirs for tool_manifest.json files
        and automatically registers the discovered tools.

        Discovery is idempotent - tools that are already registered will be
        skipped to prevent duplicates. Errors in individual tool manifests
        will not stop discovery of other tools.

        Returns:
            List of newly discovered ToolDef instances
        """
        discovered = []

        if not self.discovery_dirs:
            logger.info("No discovery directories configured")
            return discovered

        for discovery_dir in self.discovery_dirs:
            discovery_path = Path(discovery_dir)

            if not discovery_path.exists():
                logger.warning(f"Discovery directory does not exist: {discovery_path}")
                continue

            logger.info(f"Scanning directory for tools: {discovery_path}")

            # Scan for tool_manifest.json in each subdirectory
            for tool_dir in discovery_path.iterdir():
                if not tool_dir.is_dir():
                    continue

                manifest_path = tool_dir / "tool_manifest.json"
                if not manifest_path.exists():
                    logger.debug(f"No manifest found in {tool_dir.name}")
                    continue

                try:
                    # Load and parse manifest
                    tool = self._load_manifest(manifest_path)
                    if not tool:
                        continue

                    # Check if tool already registered (idempotent discovery)
                    existing = self.get(tool.tool_id)
                    if existing:
                        logger.info(f"Tool {tool.tool_id} already registered, skipping")
                        continue

                    # Register the discovered tool
                    self.register(tool)
                    discovered.append(tool)
                    logger.info(
                        f"Discovered and registered tool: {tool.tool_id} ({tool.name})"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to load tool from {manifest_path}: {e}"
                    )
                    continue

        logger.info(f"Discovery complete. Found {len(discovered)} new tools")
        return discovered

    def _load_manifest(self, manifest_path: Path) -> Optional[ToolDef]:
        """
        Load a tool definition from a manifest file.

        Args:
            manifest_path: Path to tool_manifest.json file

        Returns:
            ToolDef instance if valid, None otherwise

        Raises:
            ValueError: If manifest is invalid (handled internally, returns None)
        """
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert tool_type string to enum if present
            if 'tool_type' in data and isinstance(data['tool_type'], str):
                tool_type_str = data['tool_type'].upper()
                # Try to get enum value, default to LOCAL if not found
                try:
                    data['tool_type'] = ToolType[tool_type_str]
                except KeyError:
                    # Fallback: try by value (e.g., "local" -> ToolType.LOCAL)
                    try:
                        data['tool_type'] = ToolType(data['tool_type'].lower())
                    except ValueError:
                        logger.warning(
                            f"Unknown tool_type '{data['tool_type']}' in {manifest_path}, "
                            f"defaulting to LOCAL"
                        )
                        data['tool_type'] = ToolType.LOCAL

            tool_def = ToolDef(**data)
            return tool_def

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in manifest {manifest_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse manifest {manifest_path}: {e}")
            return None

    def get_tools_for_agent(self, tool_ids: List[str]) -> List[ToolDef]:
        """
        Get all tools required by an agent.

        Args:
            tool_ids: List of tool identifiers the agent needs

        Returns:
            List of ToolDef instances for the requested tools

        Note:
            Missing tools are logged as warnings but don't raise errors.
            This allows workflows to continue with available tools.
        """
        tools = []

        for tool_id in tool_ids:
            tool = self.get(tool_id)
            if tool:
                # Only include active tools
                if tool.status == "active":
                    tools.append(tool)
                else:
                    logger.warning(
                        f"Tool '{tool_id}' is {tool.status}, skipping"
                    )
            else:
                # Try finding by name as fallback (for frontend compatibility)
                tool = self.get_by_name(tool_id)
                if tool and tool.status == "active":
                    tools.append(tool)
                else:
                    logger.warning(f"Tool '{tool_id}' not found for agent")

        return tools

    def validate_tool_input(self, tool_id: str, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against a tool's input schema.

        Uses JSON Schema validation to check if input_data conforms to
        the tool's declared input_schema.

        Args:
            tool_id: ID of the tool to validate against
            input_data: Input data to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails with details
            KeyError: If tool not found
        """
        tool = self.get(tool_id)
        if not tool:
            raise KeyError(f"Tool '{tool_id}' not found")

        # If no input schema defined, accept any input
        if not tool.input_schema:
            return True

        try:
            # Use jsonschema for validation
            import jsonschema
            jsonschema.validate(instance=input_data, schema=tool.input_schema)
            return True

        except jsonschema.ValidationError as e:
            error_msg = f"Input validation failed for tool '{tool_id}': {e.message}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        except jsonschema.SchemaError as e:
            error_msg = f"Invalid schema for tool '{tool_id}': {e.message}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_tool_output(self, tool_id: str, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data against a tool's output schema.

        Args:
            tool_id: ID of the tool to validate against
            output_data: Output data to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
            KeyError: If tool not found
        """
        tool = self.get(tool_id)
        if not tool:
            raise KeyError(f"Tool '{tool_id}' not found")

        # If no output schema defined, accept any output
        if not tool.output_schema:
            return True

        try:
            import jsonschema
            jsonschema.validate(instance=output_data, schema=tool.output_schema)
            return True

        except jsonschema.ValidationError as e:
            error_msg = f"Output validation failed for tool '{tool_id}': {e.message}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

    def _load_all(self) -> None:
        """
        Load all tools from storage into the cache.

        Called during initialization to populate the in-memory cache.
        """
        try:
            tools = self.storage.load_all()
            for tool in tools:
                self._cache[tool.tool_id] = tool

            logger.debug(f"Loaded {len(tools)} tools into cache")

        except Exception as e:
            logger.error(f"Failed to load tools from storage: {e}")
            # Continue with empty cache rather than failing initialization

    def refresh_cache(self) -> int:
        """
        Refresh the in-memory cache from storage.

        Returns:
            Number of tools loaded
        """
        self._cache.clear()
        self._load_all()
        return len(self._cache)

    def count(self) -> int:
        """
        Get the number of registered tools.

        Returns:
            Total count of tools in registry
        """
        return len(self._cache)

    def search(self, query: str) -> List[ToolDef]:
        """
        Search tools by name or description.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching ToolDef instances
        """
        query_lower = query.lower()
        return [
            tool for tool in self._cache.values()
            if query_lower in tool.name.lower()
            or query_lower in tool.description.lower()
        ]
