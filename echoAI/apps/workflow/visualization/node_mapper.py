"""
Node Mapper: Bidirectional conversion between frontend canvas and backend workflow schema.
Handles all 16 node types with layout persistence and connection preservation.
"""
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from echolib.utils import new_id


class NodeMapper:
    """
    Bidirectional mapper for frontend canvas ↔ backend workflow schema.
    """

    # Node type to color mapping
    NODE_COLORS = {
        "Start": "#10b981",
        "End": "#64748b",
        "Agent": "#f59e0b",
        "Subagent": "#f59e0b",
        "Prompt": "#ec4899",
        "Conditional": "#8b5cf6",
        "Loop": "#8b5cf6",
        "Map": "#8b5cf6",
        "Self-Review": "#06b6d4",
        "HITL": "#06b6d4",
        "API": "#3b82f6",
        "MCP": "#3b82f6",
        "Code": "#10b981",
        "Template": "#10b981",
        "Failsafe": "#ef4444",
        "Merge": "#64748b"
    }

    # Node type to icon mapping
    NODE_ICONS = {
        "Start": "▶️",
        "End": "⏹️",
        "Agent": "🔶",
        "Subagent": "👥",
        "Prompt": "💬",
        "Conditional": "🔀",
        "Loop": "🔄",
        "Map": "🔁",
        "Self-Review": "✅",
        "HITL": "👤",
        "API": "🌐",
        "MCP": "🔌",
        "Code": "💻",
        "Template": "📝",
        "Failsafe": "🛡️",
        "Merge": "⚡"
    }

    def __init__(self, tool_registry=None):
        """
        Initialize mapper.

        Args:
            tool_registry: Tool registry for resolving tool names → IDs
        """
        self.tool_registry = tool_registry

    # ==================== FRONTEND → BACKEND ====================

    def map_frontend_to_backend(
        self,
        canvas_nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        workflow_name: Optional[str] = None,
        auto_generate_name: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Convert frontend canvas to backend workflow JSON.

        Args:
            canvas_nodes: Frontend node list
            connections: Frontend connection list
            workflow_name: Explicit workflow name (optional)
            auto_generate_name: Generate name from first agent if not provided

        Returns:
            Tuple of (workflow_dict, agents_dict)

        Raises:
            ValueError: If validation fails
        """
        # Validate Start node
        self._validate_start_node(canvas_nodes)

        # Generate workflow ID
        workflow_id = new_id("wf_")

        # Determine workflow name
        if not workflow_name and auto_generate_name:
            workflow_name = self._auto_generate_workflow_name(canvas_nodes)
        elif not workflow_name:
            workflow_name = "Untitled Workflow"

        # Convert nodes to agents
        agents_dict = {}
        agent_ids = []
        id_mapping = {}  # frontend_id → backend_agent_id

        for node in canvas_nodes:
            agent_id = new_id("agt_")
            agent = self._convert_node_to_agent(node)
            agents_dict[agent_id] = agent
            agent_ids.append(agent_id)
            id_mapping[node["id"]] = agent_id

        # Convert connections (preserve for arrow rendering)
        backend_connections = []
        for conn in connections:
            backend_connections.append({
                "id": conn.get("id", new_id("conn_")),
                "from": id_mapping[conn["from"]],
                "to": id_mapping[conn["to"]],
                "condition": conn.get("condition")  # for Conditional nodes
            })

        # Infer execution model
        execution_model = self._infer_execution_model(canvas_nodes, connections)

        # Extract state schema from Start/End nodes
        state_schema = self._extract_state_schema(canvas_nodes)

        # Extract HITL configuration
        hitl_config = self._extract_hitl(canvas_nodes, id_mapping)

        # Build hierarchy if hierarchical
        hierarchy = None
        if execution_model == "hierarchical":
            hierarchy = self._build_hierarchy(canvas_nodes, connections, id_mapping)

        # Build workflow
        workflow = {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "description": f"Created from canvas on {datetime.utcnow().strftime('%Y-%m-%d')}",
            "status": "draft",
            "version": "0.1",
            "execution_model": execution_model,
            "agents": agent_ids,
            "connections": backend_connections,
            "hierarchy": hierarchy,
            "state_schema": state_schema,
            "human_in_loop": hitl_config,
            "metadata": {
                "created_by": "workflow_builder",
                "created_at": datetime.utcnow().isoformat(),
                "canvas_layout": {
                    "width": 5000,
                    "height": 5000
                }
            }
        }

        return workflow, agents_dict

    def _validate_start_node(self, canvas_nodes: List[Dict[str, Any]]) -> None:
        """
        Validate that Start node exists and is properly positioned.

        Raises:
            ValueError: If Start node missing or invalid
        """
        start_nodes = [n for n in canvas_nodes if n.get("type") == "Start"]

        if not start_nodes:
            raise ValueError("Workflow must have a Start node")

        if len(start_nodes) > 1:
            raise ValueError("Workflow can only have one Start node")

        # Start should be the entry point (no incoming connections would be validated later)

    def _auto_generate_workflow_name(self, canvas_nodes: List[Dict[str, Any]]) -> str:
        """
        Auto-generate workflow name from first agent.

        Args:
            canvas_nodes: Frontend node list

        Returns:
            Generated workflow name
        """
        # Try to find first meaningful agent
        for node in canvas_nodes:
            if node.get("type") in ["Agent", "Subagent"]:
                agent_name = node.get("name", "Agent")
                return f"{agent_name} Workflow"

        # Fallback
        return f"Workflow {datetime.utcnow().strftime('%Y%m%d_%H%M')}"

    def _convert_node_to_agent(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert frontend node to backend agent definition.
        Handles all 16 node types.

        Args:
            node: Frontend node

        Returns:
            Backend agent definition
        """
        node_type = node.get("type", "Agent")
        config = node.get("config", {})

        # Base agent structure
        agent = {
            "agent_id": "",  # Will be set by caller
            "name": node.get("name", node_type),
            "role": self._get_node_role(node_type),
            "description": f"{node_type} node",
            "input_schema": [],
            "output_schema": [],
            "constraints": {
                "max_steps": config.get("maxIterations", 5),
                "timeout_seconds": 60
            },
            "permissions": {
                "can_call_agents": node_type == "Subagent"
            },
            "metadata": {
                "node_type": node_type,
                "ui_layout": {
                    "x": node.get("x", 300),
                    "y": node.get("y", 200),
                    "icon": node.get("icon", self.NODE_ICONS.get(node_type, "🔶")),
                    "color": node.get("color", self.NODE_COLORS.get(node_type, "#64748b"))
                },
                "created_at": datetime.utcnow().isoformat()
            }
        }

        # Node-type specific handling
        if node_type in ["Agent", "Subagent", "Prompt"]:
            agent["llm"] = self._extract_llm_config(config)
            agent["tools"] = self._resolve_tools(config.get("tools", []))

        elif node_type == "Start":
            agent["input_schema"] = [
                var.get("name") for var in config.get("inputVariables", [])
            ]

        elif node_type == "End":
            agent["output_schema"] = [
                var.get("name") for var in config.get("outputVariables", [])
            ]

        elif node_type == "Conditional":
            agent["metadata"]["branches"] = config.get("branches", [])

        elif node_type == "Loop":
            agent["metadata"]["loop_config"] = {
                "loopType": config.get("loopType", "for-each"),
                "arrayVariable": config.get("arrayVariable"),
                "maxIterations": config.get("maxIterations", 100)
            }

        elif node_type == "Map":
            agent["metadata"]["map_config"] = {
                "operation": config.get("operation"),
                "maxConcurrency": config.get("maxConcurrency", 5)
            }

        elif node_type == "HITL":
            agent["metadata"]["hitl_config"] = {
                "title": config.get("title", ""),
                "message": config.get("message", ""),
                "priority": config.get("priority", "medium"),
                "allowEdit": config.get("allowEdit", True),
                "allowDefer": config.get("allowDefer", False)
            }

        elif node_type == "API":
            agent["metadata"]["api_config"] = {
                "method": config.get("method", "GET"),
                "url": config.get("url", ""),
                "headers": config.get("headers", {}),
                "auth": config.get("auth", {})
            }

        elif node_type == "MCP":
            agent["metadata"]["mcp_config"] = {
                "serverName": config.get("serverName", ""),
                "toolName": config.get("toolName", "")
            }

        elif node_type == "Code":
            agent["metadata"]["code_config"] = {
                "language": config.get("language", "python"),
                "code": config.get("code", ""),
                "packages": config.get("packages", "")
            }

        elif node_type == "Self-Review":
            agent["metadata"]["review_config"] = {
                "checkCompleteness": config.get("checkCompleteness", True),
                "checkAccuracy": config.get("checkAccuracy", True),
                "confidenceThreshold": config.get("confidenceThreshold", 0.8)
            }

        return agent

    def _get_node_role(self, node_type: str) -> str:
        """Get role description for node type."""
        roles = {
            "Start": "Workflow entry point",
            "End": "Workflow exit point",
            "Agent": "Autonomous AI agent",
            "Subagent": "Specialist delegation",
            "Prompt": "Direct LLM call",
            "Conditional": "Conditional branching",
            "Loop": "Iteration logic",
            "Map": "Parallel execution",
            "Self-Review": "Quality validation",
            "HITL": "Human approval gate",
            "API": "HTTP request",
            "MCP": "MCP tool execution",
            "Code": "Code execution",
            "Template": "String templating",
            "Failsafe": "Error handling",
            "Merge": "Branch merging"
        }
        return roles.get(node_type, "Processing node")

    def _extract_llm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract LLM configuration from node config.
        Note: No provider field, only model.
        """
        model_config = config.get("model", {})

        return {
            "model": model_config.get("modelName", "gpt-4o-mini"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("maxTokens", 4000)
        }

    def _resolve_tools(self, frontend_tools: List[Dict[str, Any]]) -> List[str]:
        """
        Resolve tool names to tool IDs (pre-registered).

        Args:
            frontend_tools: List of frontend tool configs

        Returns:
            List of tool IDs
        """
        tool_ids = []

        for tool in frontend_tools:
            tool_name = tool.get("name", "")

            # If tool registry available, resolve name → ID
            if self.tool_registry:
                tool_id = self.tool_registry.get_tool_id_by_name(tool_name)
                if tool_id:
                    tool_ids.append(tool_id)
            else:
                # Fallback: use name as placeholder
                tool_ids.append(tool_name.lower().replace(" ", "_"))

        return tool_ids

    def _infer_execution_model(
        self,
        canvas_nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]]
    ) -> str:
        """
        Infer execution model from canvas structure.

        Returns:
            "sequential" | "parallel" | "hierarchical" | "hybrid"
        """
        node_types = [n.get("type") for n in canvas_nodes]

        # Check for hierarchical pattern (Subagent delegation)
        if "Subagent" in node_types:
            return "hierarchical"

        # Check for parallel execution
        if "Map" in node_types:
            return "parallel" if "Conditional" not in node_types else "hybrid"

        # Check for multiple branches from single node (parallel)
        outgoing_counts = {}
        for conn in connections:
            from_id = conn["from"]
            outgoing_counts[from_id] = outgoing_counts.get(from_id, 0) + 1

        has_parallel = any(count > 1 for count in outgoing_counts.values())
        has_conditional = "Conditional" in node_types or "Loop" in node_types

        if has_parallel and has_conditional:
            return "hybrid"
        elif has_parallel:
            return "parallel"
        else:
            return "sequential"

    def _extract_state_schema(self, canvas_nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract state schema from Start/End nodes."""
        state_schema = {}

        for node in canvas_nodes:
            if node.get("type") == "Start":
                for var in node.get("config", {}).get("inputVariables", []):
                    state_schema[var.get("name")] = var.get("type", "string")

            elif node.get("type") == "End":
                for var in node.get("config", {}).get("outputVariables", []):
                    state_schema[var.get("name")] = var.get("type", "string")

        return state_schema

    def _extract_hitl(
        self,
        canvas_nodes: List[Dict[str, Any]],
        id_mapping: Dict[int, str]
    ) -> Dict[str, Any]:
        """Extract human-in-the-loop configuration."""
        hitl_nodes = [n for n in canvas_nodes if n.get("type") == "HITL"]

        if not hitl_nodes:
            return {"enabled": False, "review_points": []}

        return {
            "enabled": True,
            "review_points": [id_mapping[n["id"]] for n in hitl_nodes]
        }

    def _build_hierarchy(
        self,
        canvas_nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        id_mapping: Dict[int, str]
    ) -> Optional[Dict[str, Any]]:
        """Build hierarchy structure for hierarchical workflows."""
        # Find master agent (first Subagent or first Agent)
        master_node = None
        for node in canvas_nodes:
            if node.get("type") in ["Agent", "Subagent"]:
                master_node = node
                break

        if not master_node:
            return None

        master_id = id_mapping[master_node["id"]]

        # Find all downstream agents
        downstream = []
        for conn in connections:
            if id_mapping.get(conn["from"]) == master_id:
                downstream.append(id_mapping[conn["to"]])

        return {
            "master_agent": master_id,
            "delegation_order": downstream
        }

    # ==================== BACKEND → FRONTEND ====================

    def map_backend_to_frontend(
        self,
        workflow: Dict[str, Any],
        agents_dict: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert backend workflow JSON to frontend canvas nodes.

        Args:
            workflow: Backend workflow definition
            agents_dict: Backend agents dictionary

        Returns:
            Tuple of (canvas_nodes, connections)
        """
        canvas_nodes = []
        id_mapping = {}  # backend agent_id → frontend id

        # Convert agents to canvas nodes
        for idx, agent_id in enumerate(workflow["agents"]):
            agent = agents_dict.get(agent_id)
            if not agent:
                continue

            frontend_id = idx + 1
            id_mapping[agent_id] = frontend_id

            # Extract layout (with auto-fallback)
            ui_layout = agent.get("metadata", {}).get("ui_layout", {})
            if not ui_layout:
                ui_layout = self._generate_auto_layout(idx, len(workflow["agents"]))

            node_type = agent.get("metadata", {}).get("node_type", "Agent")

            node = {
                "id": frontend_id,
                "type": node_type,
                "name": agent.get("name", node_type),
                "x": ui_layout.get("x", 300),
                "y": ui_layout.get("y", 200),
                "icon": ui_layout.get("icon", self.NODE_ICONS.get(node_type, "🔶")),
                "color": ui_layout.get("color", self.NODE_COLORS.get(node_type, "#64748b")),
                "config": self._extract_node_config(agent, node_type),
                "backend_id": agent_id,
                "status": "idle"
            }
            canvas_nodes.append(node)

        # Convert connections (preserve IDs for arrow rendering)
        connections = []
        for idx, conn in enumerate(workflow.get("connections", [])):
            from_id = id_mapping.get(conn["from"])
            to_id = id_mapping.get(conn["to"])

            if from_id and to_id:
                connections.append({
                    "id": conn.get("id", idx + 1),
                    "from": from_id,
                    "to": to_id,
                    "condition": conn.get("condition")
                })

        return canvas_nodes, connections

    def _generate_auto_layout(self, index: int, total: int) -> Dict[str, int]:
        """
        Generate auto-layout positions for nodes without ui_layout.
        Uses horizontal flow with vertical spacing.
        """
        x = 200 + (index * 250)
        y = 200 + ((index % 3) * 150)  # Stagger vertically

        return {"x": x, "y": y}

    def _extract_node_config(
        self,
        agent: Dict[str, Any],
        node_type: str
    ) -> Dict[str, Any]:
        """
        Extract frontend node config from backend agent.
        Reverse of _convert_node_to_agent.
        """
        config = {}
        metadata = agent.get("metadata", {})

        # Common configs
        if "llm" in agent:
            llm = agent["llm"]
            config["model"] = {
                "modelName": llm.get("model", "gpt-4o-mini"),
                "displayName": llm.get("model", "GPT-4o Mini")
            }
            config["temperature"] = llm.get("temperature", 0.7)
            config["maxTokens"] = llm.get("max_tokens", 4000)
            config["tools"] = self._tools_ids_to_names(agent.get("tools", []))

        # Node-type specific
        if node_type == "Start":
            config["inputVariables"] = [
                {"name": name, "type": "string", "required": True}
                for name in agent.get("input_schema", [])
            ]

        elif node_type == "End":
            config["outputVariables"] = [
                {"name": name, "type": "string"}
                for name in agent.get("output_schema", [])
            ]

        elif node_type == "Conditional":
            config["branches"] = metadata.get("branches", [])

        elif node_type == "Loop":
            config.update(metadata.get("loop_config", {}))

        elif node_type == "Map":
            config.update(metadata.get("map_config", {}))

        elif node_type == "HITL":
            config.update(metadata.get("hitl_config", {}))

        elif node_type == "API":
            config.update(metadata.get("api_config", {}))

        elif node_type == "MCP":
            config.update(metadata.get("mcp_config", {}))

        elif node_type == "Code":
            config.update(metadata.get("code_config", {}))

        elif node_type == "Self-Review":
            config.update(metadata.get("review_config", {}))

        return config

    def _tools_ids_to_names(self, tool_ids: List[str]) -> List[Dict[str, Any]]:
        """Convert tool IDs back to frontend tool configs."""
        tools = []
        for tool_id in tool_ids:
            tools.append({
                "id": new_id("tool_"),
                "name": tool_id.replace("_", " ").title(),
                "type": "tools",
                "enabled": True,
                "config": {}
            })
        return tools
