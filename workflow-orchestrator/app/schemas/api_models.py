"""
API Request/Response Models for Workflow Orchestrator
"""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# META-PROMPT GENERATOR MODELS
# ============================================================================

class UserRequest(BaseModel):
    """User's natural language request for workflow generation"""
    request: str = Field(..., min_length=10, description="Natural language description of the workflow needed")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context or constraints")


class DomainAnalysis(BaseModel):
    """Structured analysis of user request"""
    original_request: str
    domain: str = Field(..., description="Primary domain (inventory, sales, etc.)")
    entities: List[str] = Field(..., description="Key entities involved")
    operations: List[str] = Field(..., description="Required operations")
    data_sources: List[str] = Field(..., description="Data sources needed")
    output_requirements: List[str] = Field(..., description="Expected outputs")
    temporal_requirements: Optional[str] = Field(default="on-demand", description="Temporal requirements")
    integration_points: List[str] = Field(default_factory=list, description="External integrations needed")
    suggested_patterns: List[str] = Field(..., description="Suggested communication patterns")
    complexity_score: int = Field(..., ge=1, le=10, description="System complexity score")


class MetaPromptResponse(BaseModel):
    """Response from meta-prompt generation"""
    analysis: DomainAnalysis
    meta_prompt: str = Field(..., description="Structured prompt for designer LLM")


# ============================================================================
# AGENT SYSTEM DESIGNER MODELS
# ============================================================================

class LLMConfig(BaseModel):
    """LLM configuration for an agent"""
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(..., ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=100, le=128000)
    top_p: float = Field(default=1.0, ge=0, le=1)


class AgentPermissions(BaseModel):
    """Agent permission configuration"""
    can_delegate: bool = Field(default=False, description="Can this agent delegate to other agents")
    can_call_agents: Optional[List[str]] = Field(default=None, description="List of agent IDs this agent can call")
    max_tool_calls: int = Field(default=50, ge=1, le=1000, description="Maximum tool calls allowed")


class AgentDefinition(BaseModel):
    """Complete agent specification"""
    id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", description="Unique agent ID")
    name: str = Field(..., description="Human-readable name")
    role: str = Field(..., description="Agent role")
    responsibilities: List[str] = Field(..., min_length=1, max_length=10)
    system_prompt: str = Field(..., min_length=50, description="Detailed LLM instructions")
    tools: List[str] = Field(default_factory=list, description="Tool names this agent uses")
    llm_config: LLMConfig
    permissions: AgentPermissions = Field(default_factory=AgentPermissions, description="Agent permissions")
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    is_master: bool = Field(default=False, description="Is this a master/coordinator agent")


class ToolDefinition(BaseModel):
    """Tool specification"""
    name: str = Field(..., description="Unique tool name")
    type: Literal["database", "api", "calculator", "email", "search", "file", "custom"]
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    auth_required: bool = Field(default=False)
    parameters: Optional[Dict[str, Any]] = None


class WorkflowStep(BaseModel):
    """Single step in a workflow"""
    agent_id: str = Field(..., description="Agent executing this step")
    action: str = Field(..., description="Action to perform")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None
    parallel_with: List[str] = Field(default_factory=list)


class WorkflowDefinition(BaseModel):
    """Complete workflow specification"""
    name: str = Field(..., description="Unique workflow name")
    description: str
    trigger: Literal["manual", "scheduled", "event", "webhook"] = Field(default="manual")
    steps: List[WorkflowStep] = Field(..., min_length=1)
    communication_pattern: Literal["sequential", "parallel", "hierarchical", "conditional", "graph"]


class AgentSystemDesign(BaseModel):
    """Complete agent system design from Designer LLM"""
    system_name: str
    description: str
    domain: str
    agents: List[AgentDefinition] = Field(..., min_length=1)
    tools: List[ToolDefinition] = Field(default_factory=list)
    workflows: List[WorkflowDefinition] = Field(..., min_length=1)
    communication_pattern: Literal["sequential", "parallel", "hierarchical", "conditional", "graph"]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationError(BaseModel):
    """Single validation error"""
    severity: Literal["error", "warning", "info"]
    location: str = Field(..., description="Where the error occurred")
    message: str
    suggestion: Optional[str] = None


class ValidationResponse(BaseModel):
    """Result of validation"""
    valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# STORAGE MODELS
# ============================================================================

class WorkflowState(BaseModel):
    """Workflow lifecycle state"""
    state: Literal["draft", "temp", "final"]
    version: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SaveWorkflowRequest(BaseModel):
    """Request to save workflow"""
    workflow_id: str
    agent_system: AgentSystemDesign
    state: Literal["draft", "temp", "final"]
    version: Optional[str] = None


class SaveWorkflowResponse(BaseModel):
    """Response from save operation"""
    success: bool
    workflow_id: str
    state: str
    version: Optional[str] = None
    file_path: str


class CloneWorkflowRequest(BaseModel):
    """Request to clone FINAL → DRAFT"""
    workflow_id: str
    from_version: str


# ============================================================================
# EXECUTION MODELS
# ============================================================================

class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow"""
    workflow_id: str
    execution_mode: Literal["test", "final"]
    version: Optional[str] = None
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    thread_id: Optional[str] = None


class ExecutionStatus(BaseModel):
    """Status of workflow execution"""
    run_id: str
    workflow_id: str
    status: Literal["running", "completed", "failed", "paused"]
    current_step: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    thread_id: Optional[str] = Field(default=None, description="Thread ID for conversation continuation and HITL")


# ============================================================================
# API RESPONSE WRAPPERS
# ============================================================================

class DesignWorkflowRequest(BaseModel):
    """Complete request to design a workflow from user input"""
    user_request: UserRequest


class DesignWorkflowResponse(BaseModel):
    """Complete response with designed agent system"""
    analysis: DomainAnalysis
    agent_system: AgentSystemDesign
    meta_prompt_used: str


# ============================================================================
# SERVICE LAYER DTOs (for internal service-to-service communication)
# ============================================================================

# --- Validator Service DTOs ---

class ValidateAgentSystemRequest(BaseModel):
    """Request for agent system validation"""
    agent_system: AgentSystemDesign
    mode: Literal["draft", "temp", "final"] = "draft"


class ValidateAgentRequest(BaseModel):
    """Request for single agent validation"""
    agent: AgentDefinition


class ValidateWorkflowRequest(BaseModel):
    """Request for workflow validation"""
    workflow: WorkflowDefinition


# --- Workflow Service DTOs ---

class CompileWorkflowRequest(BaseModel):
    """Request to compile workflow JSON to LangGraph"""
    agent_system: AgentSystemDesign
    workflow_name: Optional[str] = None


class CompileWorkflowResponse(BaseModel):
    """Response from workflow compilation"""
    success: bool
    workflow_name: str
    graph_compiled: bool
    error: Optional[str] = None


class ModifyWorkflowRequest(BaseModel):
    """Request to modify existing workflow (HITL)"""
    agent_system: AgentSystemDesign
    modification_request: str


class VersionWorkflowRequest(BaseModel):
    """Request to bump workflow version"""
    workflow_id: str
    current_version: str
    bump_type: Literal["major", "minor"] = "minor"


class VersionWorkflowResponse(BaseModel):
    """Response from version bump"""
    workflow_id: str
    old_version: str
    new_version: str


# --- Agent Service DTOs ---

class CreateAgentRequest(BaseModel):
    """Request to create runtime agent instance"""
    agent_definition: AgentDefinition


class CreateAgentResponse(BaseModel):
    """Response from agent creation"""
    agent_id: str
    created: bool
    error: Optional[str] = None


class ValidateAgentPermissionsRequest(BaseModel):
    """Request to validate agent permissions"""
    agent_id: str
    target_agent_id: str
    action: Literal["call", "delegate", "read_state", "write_state"]


class ValidateAgentPermissionsResponse(BaseModel):
    """Response from permission validation"""
    allowed: bool
    reason: Optional[str] = None


class ListAgentsResponse(BaseModel):
    """Response listing all agents"""
    agents: List[AgentDefinition]
    count: int


# --- Runtime Service DTOs ---

class RuntimeExecuteRequest(BaseModel):
    """Request for runtime execution"""
    workflow_id: str
    execution_mode: Literal["test", "final"]
    version: Optional[str] = None
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    thread_id: Optional[str] = None
    enable_hitl: bool = True


class RuntimeResumeRequest(BaseModel):
    """Request to resume paused execution"""
    workflow_id: str
    thread_id: str
    human_decision: Dict[str, Any]


class RuntimeMetrics(BaseModel):
    """Runtime execution metrics"""
    run_id: str
    total_tokens: int
    total_cost_inr: float
    duration_seconds: float
    steps_executed: int
    agents_called: Dict[str, int]


class CheckpointInfo(BaseModel):
    """Checkpoint information"""
    thread_id: str
    checkpoint_id: str
    agent_id: str
    state_snapshot: Dict[str, Any]
    created_at: datetime


# --- Storage Service DTOs ---

class LoadWorkflowRequest(BaseModel):
    """Request to load workflow from storage"""
    workflow_id: str
    state: Literal["draft", "temp", "final"]
    version: Optional[str] = None


class ListWorkflowsResponse(BaseModel):
    """Response listing workflows"""
    workflows: List[Dict[str, Any]]
    count: int


class DeleteWorkflowRequest(BaseModel):
    """Request to delete workflow"""
    workflow_id: str
    state: Literal["draft", "temp"]


class ArchiveWorkflowRequest(BaseModel):
    """Request to archive FINAL workflow"""
    workflow_id: str
    version: str


# --- Visualization Service DTOs ---

class GraphNode(BaseModel):
    """Node in workflow graph"""
    id: str
    label: str
    type: Literal["agent", "master_agent", "tool", "start", "end"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Edge in workflow graph"""
    source: str
    target: str
    label: Optional[str] = None
    condition: Optional[str] = None


class WorkflowGraphRequest(BaseModel):
    """Request to generate workflow graph"""
    agent_system: AgentSystemDesign
    workflow_name: Optional[str] = None


class WorkflowGraphResponse(BaseModel):
    """Response with workflow graph"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    layout: Dict[str, Any] = Field(default_factory=dict)


class ApplyGraphEditRequest(BaseModel):
    """Request to apply UI graph edits back to workflow"""
    agent_system: AgentSystemDesign
    graph_edits: Dict[str, Any]


class ApplyGraphEditResponse(BaseModel):
    """Response after applying graph edits"""
    updated_agent_system: AgentSystemDesign
    changes_applied: List[str]
    validation_required: bool


# --- Telemetry Service DTOs ---

class TelemetryQuery(BaseModel):
    """Query for telemetry data"""
    workflow_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ExecutionMetrics(BaseModel):
    """Aggregated execution metrics"""
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_duration_seconds: float
    total_cost_inr: float
    total_tokens: int


class AgentMetrics(BaseModel):
    """Agent-specific metrics"""
    agent_id: str
    invocations: int
    avg_duration_seconds: float
    total_cost_inr: float
    success_rate: float


class TelemetryResponse(BaseModel):
    """Response with telemetry data"""
    execution_metrics: Optional[ExecutionMetrics] = None
    agent_metrics: List[AgentMetrics] = Field(default_factory=list)
    spans: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# UNIFIED PROCESS ENDPOINT MODELS (NEW - does not modify existing)
# ============================================================================

class ProcessMessageRequest(BaseModel):
    """
    Unified request for processing user messages.
    Backend handles intent detection and routing.
    """
    message: str = Field(..., min_length=1, description="User's message/instruction")
    workflow_id: Optional[str] = Field(default=None, description="Current workflow ID if exists")
    agent_system: Optional[Dict[str, Any]] = Field(default=None, description="Current workflow data if exists")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for conversation continuation")
    execution_mode: Optional[Literal["test", "final"]] = Field(default="test", description="Execution mode")
    pending_modification: bool = Field(default=False, description="Force treat as modification request")


class ProcessMessageResponse(BaseModel):
    """
    Unified response from process endpoint.
    Contains action taken and result.
    """
    action: Literal["generate", "modify", "test", "execute", "save", "error"] = Field(..., description="Action that was performed")
    intent_detected: str = Field(..., description="Intent detected from message")
    success: bool = Field(default=True, description="Whether action succeeded")

    # Workflow data (for generate/modify actions)
    agent_system: Optional[Dict[str, Any]] = Field(default=None, description="Workflow data after action")

    # Execution data (for test/execute actions)
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result if tested/executed")
    run_id: Optional[str] = Field(default=None, description="Run ID if executed")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for conversation continuation")

    # Validation data
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Validation result if validated")

    # Error info
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # Human readable message
    message: str = Field(default="", description="Human readable response message")
