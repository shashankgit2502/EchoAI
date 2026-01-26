from enum import Enum

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional

class Event(BaseModel):
    type: str
    data: Dict[str, Any]

class Session(BaseModel):
    id: str
    user_id: str
    data: Dict[str, Any] = {}

class UserContext(BaseModel):
    user_id: str
    email: str

class Document(BaseModel):
    id: str
    title: str
    content: str

class IndexSummary(BaseModel):
    count: int

class ContextBundle(BaseModel):
    documents: List[Document]

class LLMOutput(BaseModel):
    text: str
    tokens: int = 0


# ==================== TOOL SYSTEM TYPES ====================

class ToolType(str, Enum):
    """
    Classification of tool execution types.

    LOCAL: Python function in tools folder (executed locally)
    MCP: MCP connector endpoint (executed via MCP protocol)
    API: Direct HTTP API call (executed via HTTP request)
    CREWAI: CrewAI-native tool (executed within CrewAI context)
    """
    LOCAL = "local"
    MCP = "mcp"
    API = "api"
    CREWAI = "crewai"


class ToolDef(BaseModel):
    """
    Complete tool definition with execution configuration.

    This model defines everything needed to register, discover, validate,
    and execute a tool within the EchoAI system.

    Attributes:
        tool_id: Unique identifier (e.g., "tool_calculator", "tool_web_search")
        name: Human-readable tool name
        description: Detailed description of tool functionality
        tool_type: How the tool is executed (LOCAL, MCP, API, CREWAI)
        input_schema: JSON Schema for validating tool input
        output_schema: JSON Schema for validating tool output
        execution_config: Type-specific configuration for tool execution
        version: Semantic version of the tool
        tags: Categorization tags for discovery and filtering
        status: Current status (active, deprecated, disabled)
        metadata: Additional metadata (author, created_at, etc.)
    """
    # Required identification fields
    tool_id: str = Field(default="", description="Unique tool identifier")
    name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Tool functionality description")

    # Tool type and configuration
    tool_type: ToolType = Field(default=ToolType.LOCAL, description="Execution type")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for input validation"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for output validation"
    )
    execution_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific execution configuration"
    )

    # Metadata fields
    version: str = Field(default="1.0", description="Tool version")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    status: str = Field(default="active", description="Tool status")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator('tool_id', mode='before')
    @classmethod
    def ensure_tool_id(cls, v: str, info) -> str:
        """Generate tool_id from name if not provided."""
        if not v and info.data.get('name'):
            # Generate tool_id from name: "Calculator" -> "tool_calculator"
            name = info.data['name']
            return f"tool_{name.lower().replace(' ', '_').replace('-', '_')}"
        return v

    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Ensure status is one of the allowed values."""
        allowed = {"active", "deprecated", "disabled"}
        if v.lower() not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator('input_schema', 'output_schema')
    @classmethod
    def validate_schema_structure(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that schema has basic JSON Schema structure if provided."""
        if v and 'type' not in v:
            # Add default type if schema has content but no type
            v['type'] = 'object'
        return v


class ToolRef(BaseModel):
    """Lightweight reference to a registered tool."""
    name: str
    tool_id: Optional[str] = None


class ToolResult(BaseModel):
    """
    Result of a tool execution.

    Attributes:
        name: Tool name that was executed
        tool_id: Tool identifier
        output: Execution output data
        success: Whether execution succeeded
        error: Error message if execution failed
        metadata: Execution metadata (timing, context, etc.)
    """
    name: str = Field(..., description="Tool name")
    tool_id: str = Field(default="", description="Tool identifier")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    success: bool = Field(default=True, description="Execution success flag")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata"
    )

class AppDef(BaseModel):
    name: str
    config: Dict[str, Any]

class App(BaseModel):
    id: str
    name: str
    config: Dict[str, Any]

class DeployResult(BaseModel):
    app_id: str
    env: str
    status: str

class AgentTemplate(BaseModel):
    name: str
    icon: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    variables: Optional[List[Dict[str, Any]]] = None
    settings: Optional[Dict[str, Any]] = None
    source: Optional[str] = None  # "template", "llm_generated", "user"

# Enhanced LLM Configuration
class LLMConfig(BaseModel):
    provider: str  # openai, anthropic, azure, local
    model: str
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None

# Enhanced Agent with full orchestrator support
class Agent(BaseModel):
    id: str
    name: str
    # NEW FIELDS (optional for backward compatibility)
    role: Optional[str] = None
    description: Optional[str] = None
    llm: Optional[LLMConfig] = None
    tools: Optional[List[str]] = None  # Tool IDs
    input_schema: Optional[List[str]] = None
    output_schema: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None
    permissions: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Workflow Connection (for graph topology)
class WorkflowConnection(BaseModel):
    from_agent: str
    to_agent: str
    condition: Optional[str] = None

# Workflow Hierarchy (for hierarchical execution)
class WorkflowHierarchy(BaseModel):
    master_agent: str
    delegation_order: List[str]

# HITL Configuration
class HITLConfig(BaseModel):
    enabled: bool = False
    review_points: Optional[List[str]] = None

# Workflow Validation Info
class WorkflowValidation(BaseModel):
    validated_by: Optional[str] = None
    validated_at: Optional[str] = None
    validation_hash: Optional[str] = None

# Enhanced Workflow with full orchestrator support
class Workflow(BaseModel):
    id: str
    name: str
    # NEW FIELDS (optional for backward compatibility)
    description: Optional[str] = None
    status: Optional[str] = "draft"  # draft, validated, testing, final
    version: Optional[str] = "0.1"
    execution_model: Optional[str] = None  # sequential, parallel, hierarchical, hybrid
    agents: Optional[List[str]] = None  # Agent IDs
    connections: Optional[List[WorkflowConnection]] = None
    hierarchy: Optional[WorkflowHierarchy] = None
    state_schema: Optional[Dict[str, str]] = None  # key -> type
    human_in_loop: Optional[HITLConfig] = None
    validation: Optional[WorkflowValidation] = None
    metadata: Optional[Dict[str, Any]] = None

# Enhanced Validation Result
class ValidationResult(BaseModel):
    ok: bool
    details: Optional[str] = None
    # NEW FIELDS
    valid: Optional[bool] = None  # Alias for ok
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

class ConnectorDef(BaseModel):
    name: str
    config: Dict[str, Any]

class ConnectorRef(BaseModel):
    name: str

class ConnectorResult(BaseModel):
    name: str
    result: Dict[str, Any]

class Health(BaseModel):
    status: str

# ==================== ORCHESTRATOR TYPES ====================

# Graph Visualization
class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # agent, master_agent, start, end
    metadata: Optional[Dict[str, Any]] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    condition: Optional[str] = None

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

# Workflow Execution
class ExecuteWorkflowRequest(BaseModel):
    workflow_id: str
    mode: str  # "test" or "final"
    version: Optional[str] = None
    input_payload: Optional[Dict[str, Any]] = {}

class ExecutionResponse(BaseModel):
    run_id: str
    status: str
    output: Optional[Dict[str, Any]] = None

# Workflow Lifecycle
class WorkflowValidationRequest(BaseModel):
    workflow: Dict[str, Any]
    agents: Dict[str, Dict[str, Any]]

class SaveFinalRequest(BaseModel):
    workflow: Dict[str, Any]

class CloneWorkflowRequest(BaseModel):
    workflow_id: str
    from_version: str

# MCP Tool Definition (Enhanced)
class MCPToolConfig(BaseModel):
    server: str  # MCP server name or URL
    endpoint: str  # MCP endpoint
    version: Optional[str] = None

class MCPToolDefinition(BaseModel):
    tool_id: str
    name: str
    description: str
    mcp: MCPToolConfig
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    permissions: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    status: Optional[str] = "active"  # active, deprecated
    metadata: Optional[Dict[str, Any]] = None

# Workflow Metrics (Telemetry)
class WorkflowMetrics(BaseModel):
    workflow_id: str
    version: str
    total_duration_ms: float
    agent_metrics: Dict[str, Dict[str, Any]]
