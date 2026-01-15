
from pydantic import BaseModel
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

class ToolDef(BaseModel):
    name: str
    description: str

class ToolRef(BaseModel):
    name: str

class ToolResult(BaseModel):
    name: str
    output: Dict[str, Any]

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
