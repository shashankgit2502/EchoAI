"""
Core Constants and Enums
Central source of truth for workflow states, execution modes, and system limits
"""
from enum import Enum


# ============================================================================
# WORKFLOW LIFECYCLE STATES
# ============================================================================

class WorkflowState(str, Enum):
    """
    Workflow lifecycle states

    DRAFT: Editable, not validated, for design-time iteration
    TEMP: Validated, for testing, overwrites allowed
    FINAL: Immutable, versioned, production-ready
    ARCHIVED: Historical versions moved to archive
    """
    DRAFT = "draft"
    TEMP = "temp"
    FINAL = "final"
    ARCHIVED = "archived"


# ============================================================================
# EXECUTION MODES
# ============================================================================

class ExecutionMode(str, Enum):
    """
    Workflow execution modes

    TEST: Execute TEMP workflow (sandbox mode)
    FINAL: Execute FINAL workflow (production mode)
    """
    TEST = "test"
    FINAL = "final"


class ExecutionStatus(str, Enum):
    """
    Workflow execution status

    PENDING: Execution queued but not started
    RUNNING: Currently executing
    PAUSED: Paused for human-in-the-loop (HITL)
    COMPLETED: Successfully completed
    FAILED: Execution failed with error
    CANCELLED: Manually cancelled by user
    """
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# COMMUNICATION PATTERNS
# ============================================================================

class CommunicationPattern(str, Enum):
    """
    Agent communication patterns

    SEQUENTIAL: A → B → C (linear pipeline)
    PARALLEL: A, B, C execute concurrently, synchronize at end
    HIERARCHICAL: Master agent delegates to workers, aggregates results
    CONDITIONAL: Branching logic based on runtime conditions
    GRAPH: Arbitrary DAG topology with explicit connections
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONDITIONAL = "conditional"
    GRAPH = "graph"


# ============================================================================
# AGENT ROLES
# ============================================================================

class AgentRole(str, Enum):
    """
    Common agent role categories
    """
    COORDINATOR = "coordinator"
    ANALYZER = "analyzer"
    PROCESSOR = "processor"
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator"
    SPECIALIST = "specialist"
    GATEWAY = "gateway"


# ============================================================================
# TOOL TYPES
# ============================================================================

class ToolType(str, Enum):
    """
    Supported tool types
    """
    DATABASE = "database"
    API = "api"
    CALCULATOR = "calculator"
    EMAIL = "email"
    SEARCH = "search"
    FILE = "file"
    CUSTOM = "custom"
    MCP = "mcp"  # Model Context Protocol tools


# ============================================================================
# VALIDATION SEVERITY
# ============================================================================

class ValidationSeverity(str, Enum):
    """
    Validation error severity levels

    ERROR: Blocks progression (workflow cannot advance)
    WARNING: Advisory (workflow can advance with caution)
    INFO: Informational (best practice suggestion)
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================================
# SYSTEM LIMITS
# ============================================================================

class SystemLimits:
    """
    System-wide limits to prevent abuse and ensure stability
    """
    # Workflow limits
    MAX_AGENTS_PER_WORKFLOW = 50
    MAX_TOOLS_PER_AGENT = 20
    MAX_WORKFLOW_STEPS = 100
    MAX_WORKFLOW_NAME_LENGTH = 100
    MAX_WORKFLOW_DESCRIPTION_LENGTH = 500

    # Agent limits
    MAX_AGENT_ID_LENGTH = 50
    MIN_SYSTEM_PROMPT_LENGTH = 50
    MAX_SYSTEM_PROMPT_LENGTH = 10000
    MAX_AGENT_RESPONSIBILITIES = 10

    # Execution limits
    DEFAULT_EXECUTION_TIMEOUT_SECONDS = 300  # 5 minutes
    MAX_EXECUTION_TIMEOUT_SECONDS = 3600  # 1 hour
    MAX_RETRY_ATTEMPTS = 3
    MAX_CONCURRENT_EXECUTIONS = 10
    MAX_CHECKPOINT_SIZE_MB = 100

    # LLM limits
    MIN_LLM_TEMPERATURE = 0.0
    MAX_LLM_TEMPERATURE = 2.0
    MIN_LLM_MAX_TOKENS = 100
    MAX_LLM_MAX_TOKENS = 128000

    # Storage limits
    MAX_WORKFLOW_VERSIONS = 50
    MAX_ARCHIVE_AGE_DAYS = 365
    MAX_TEMP_WORKFLOW_AGE_HOURS = 24

    # API limits
    MAX_BATCH_SIZE = 10
    MAX_HISTORY_LIMIT = 1000
    DEFAULT_HISTORY_LIMIT = 100


# ============================================================================
# LLM MODELS
# ============================================================================

class LLMModel(str, Enum):
    """
    Supported LLM models
    """
    # Anthropic Claude
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU_3_5 = "claude-3-5-haiku-20241022"

    # OpenAI GPT
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # For extensibility
    CUSTOM = "custom"


# ============================================================================
# FILE EXTENSIONS
# ============================================================================

class FileExtension:
    """
    Standard file extensions used in the system
    """
    DRAFT = ".draft.json"
    TEMP = ".temp.json"
    FINAL = ".final.json"
    AGENT = ".json"
    TOOL = ".json"
    CHECKPOINT = ".checkpoint.json"


# ============================================================================
# API RESPONSE CODES
# ============================================================================

class ResponseMessage:
    """
    Standard API response messages
    """
    # Success
    WORKFLOW_CREATED = "Workflow created successfully"
    WORKFLOW_UPDATED = "Workflow updated successfully"
    WORKFLOW_SAVED = "Workflow saved successfully"
    WORKFLOW_DELETED = "Workflow deleted successfully"
    WORKFLOW_CLONED = "Workflow cloned successfully"

    # Validation
    VALIDATION_PASSED = "Validation passed successfully"
    VALIDATION_FAILED = "Validation failed"

    # Execution
    EXECUTION_STARTED = "Workflow execution started"
    EXECUTION_COMPLETED = "Workflow execution completed"
    EXECUTION_FAILED = "Workflow execution failed"
    EXECUTION_PAUSED = "Workflow execution paused for HITL"

    # Errors
    WORKFLOW_NOT_FOUND = "Workflow not found"
    AGENT_NOT_FOUND = "Agent not found"
    TOOL_NOT_FOUND = "Tool not found"
    INVALID_STATE_TRANSITION = "Invalid workflow state transition"
    IMMUTABLE_WORKFLOW = "Cannot modify FINAL workflow (clone to DRAFT first)"
    VERSION_ALREADY_EXISTS = "Version already exists"


# ============================================================================
# TELEMETRY
# ============================================================================

class TelemetrySpan:
    """
    OpenTelemetry span names
    """
    # Workflow operations
    WORKFLOW_DESIGN = "workflow.design"
    WORKFLOW_VALIDATE = "workflow.validate"
    WORKFLOW_SAVE = "workflow.save"
    WORKFLOW_LOAD = "workflow.load"
    WORKFLOW_COMPILE = "workflow.compile"

    # Execution operations
    WORKFLOW_EXECUTE = "workflow.execute"
    AGENT_EXECUTE = "agent.execute"
    TOOL_EXECUTE = "tool.execute"

    # HITL operations
    WORKFLOW_PAUSE = "workflow.pause"
    WORKFLOW_RESUME = "workflow.resume"


class TelemetryAttribute:
    """
    Standard OpenTelemetry attribute keys
    """
    # Workflow attributes
    WORKFLOW_ID = "workflow.id"
    WORKFLOW_NAME = "workflow.name"
    WORKFLOW_STATE = "workflow.state"
    WORKFLOW_VERSION = "workflow.version"
    WORKFLOW_PATTERN = "workflow.pattern"

    # Agent attributes
    AGENT_ID = "agent.id"
    AGENT_ROLE = "agent.role"
    AGENT_MODEL = "agent.model"

    # Execution attributes
    RUN_ID = "execution.run_id"
    THREAD_ID = "execution.thread_id"
    EXECUTION_MODE = "execution.mode"
    EXECUTION_STATUS = "execution.status"

    # Performance attributes
    TOKEN_COUNT = "llm.token_count"
    LATENCY_MS = "latency.ms"
    COST_USD = "cost.usd"


# ============================================================================
# ERROR CODES
# ============================================================================

class ErrorCode:
    """
    System error codes for debugging and monitoring
    """
    # Validation errors (1xxx)
    SCHEMA_VALIDATION_ERROR = "E1001"
    TOPOLOGY_VALIDATION_ERROR = "E1002"
    TOOL_REFERENCE_ERROR = "E1003"
    AGENT_REFERENCE_ERROR = "E1004"
    CIRCULAR_DEPENDENCY_ERROR = "E1005"

    # Storage errors (2xxx)
    STORAGE_READ_ERROR = "E2001"
    STORAGE_WRITE_ERROR = "E2002"
    WORKFLOW_NOT_FOUND = "E2003"
    VERSION_CONFLICT = "E2004"
    IMMUTABLE_VIOLATION = "E2005"

    # Execution errors (3xxx)
    COMPILATION_ERROR = "E3001"
    EXECUTION_ERROR = "E3002"
    TIMEOUT_ERROR = "E3003"
    CHECKPOINT_ERROR = "E3004"
    HITL_ERROR = "E3005"

    # LLM errors (4xxx)
    LLM_UNAVAILABLE = "E4001"
    LLM_QUOTA_EXCEEDED = "E4002"
    LLM_INVALID_RESPONSE = "E4003"

    # Tool errors (5xxx)
    TOOL_UNAVAILABLE = "E5001"
    TOOL_EXECUTION_ERROR = "E5002"
    MCP_CONNECTION_ERROR = "E5003"
