
import json
from pathlib import Path
from pydantic import BaseModel, Field
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
    """
    AgentTemplate loads and manages the agent.json template file.
    Returns the template as JSON/JSON object to be consumed by AgentService.
    """
    template_path: Optional[Path] = None
    template_data: Optional[Dict[str, Any]] = None
    
    def __init__(self, name: Optional[str] = None, template_path: Optional[str] = None, **kwargs):
        """
        Initialize AgentTemplate.
        
        Args:
            name: Optional agent name (for backward compatibility)
            template_path: Optional path to template file. Defaults to backend/data/templates/agent.json
        """
        # Determine template file path
        if template_path:
            path = Path(template_path)
        else:
            # Default to backend/data/templates/agent.json relative to echolib/types.py
            current_file = Path(__file__)  # backend/echolib/types.py
            path = current_file.parent.parent / "data" / "templates" / "agent.json"  # backend/data/templates/agent.json
        
        # Load the template JSON
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        # Update name if provided
        if name and template_data:
            template_data['name'] = name
        
        # Initialize with Pydantic
        super().__init__(template_path=path, template_data=template_data, **kwargs)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Returns the template as a JSON-compatible dictionary.
        This can be consumed by the AgentService to update all Agent fields.
        
        Returns:
            Dict[str, Any]: The complete template as a dictionary
        """
        if self.template_data is None:
            raise ValueError("Template data not loaded")
        return self.template_data.copy()
    
    def to_json_string(self, indent: int = 2) -> str:
        """
        Returns the template as a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            str: The template as a formatted JSON string
        """
        return json.dumps(self.to_json(), indent=indent)
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'AgentTemplate':
        """
        Create an AgentTemplate instance from a JSON dictionary.
        
        Args:
            json_data: Dictionary containing the template data
            
        Returns:
            AgentTemplate: Instance with the provided template data
        """
        instance = cls.__new__(cls)
        instance.template_path = None
        instance.template_data = json_data.copy()
        super(AgentTemplate, instance).__init__(template_path=None, template_data=instance.template_data)
        return instance

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
    input_schema: Optional[List[Dict[str, Any]]] = None
    output_schema: Optional[List[Dict[str, Any]]] = None
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


class CardCreate(BaseModel):
    agent_name: str = Field(..., min_length=1, description="Human-readable name of the agent")
    purpose: str = Field(..., min_length=1, description="What this agent is for")
    goals: Optional[List[str]] = Field(default=None, description="High-level goals (optional)")
    input_assumptions: Optional[str] = Field(default=None, description="Assumptions about inputs (optional)")
    output_definitions: Optional[str] = Field(default=None, description="Expected outputs / format (optional)")
    tools_apis_used: List[str] = Field(default_factory=list, description="Tools/APIs the agent uses")
    reasoning_working_style: Optional[str] = Field(default=None, description="How the agent reasons/works (optional)")
    error_handling_patterns: Optional[str] = Field(default=None, description="How to handle errors (optional)")
    example_workflows: Optional[List[str]] = Field(default=None, description="Example workflows (optional)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "agent_name": "Procurement Helper",
                "purpose": "Automate vendor onboarding and RFQ triage",
                "goals": [
                    "Reduce average onboarding time by 30%",
                    "Ensure RFQs are classified and routed correctly"
                ],
                "input_assumptions": "CSV uploads or SharePoint folders; user provides cost center",
                "output_definitions": "JSON summary plus XLSX export; posts status to Teams channel",
                "tools_apis_used": ["SharePoint", "SAP RFC", "OpenAI Embeddings"],
                "reasoning_working_style": "Reason step-by-step; prefer structured outputs",
                "error_handling_patterns": "Retry transient network errors up to 3 times; escalate with ticket",
                "example_workflows": [
                    "User uploads vendor CSV → validate → create SAP vendor → send summary",
                    "Classify RFQ email → extract entities → route to commodity owner"
                ]
            }
        }
    }

class CardResponse(BaseModel):
    id: str
    agent_name: str
    purpose: str
    status: str = "created"
